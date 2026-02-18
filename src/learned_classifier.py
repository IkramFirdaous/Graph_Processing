"""
Learned Classifier - Classification par Random Forest

Remplace les regles heuristiques par un classificateur appris
(Random Forest + PCA sur descripteurs HOG/LBP/ORB).

Concepts CV :
- Reconnaissance (Theme 6) : classification supervisee
- Random Forest : ensemble de decision trees
- PCA : reduction de dimensionnalite
- Fallback sur heuristique si pas de modele entraine

Usage:
    # Entrainement
    classifier = LearnedClassifier()
    classifier.train(images_gray, labels)
    classifier.save()

    # Prediction
    classifier = LearnedClassifier()
    classifier.load()
    diagram_type, confidence = classifier.predict(img_gray)
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

try:
    from .feature_descriptors import FeatureDescriptorExtractor, FeatureVector
    from .image_classifier import ImageTypeClassifier, classify_image_type
except ImportError:
    from feature_descriptors import FeatureDescriptorExtractor, FeatureVector
    from image_classifier import ImageTypeClassifier, classify_image_type


DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


class LearnedClassifier:
    """
    Classificateur appris (Random Forest + PCA)

    Pipeline :
    1. Extraction de features (HOG + LBP + ORB + couleur + geometrique)
    2. StandardScaler pour normalisation
    3. PCA pour reduction de dimensionnalite
    4. Random Forest pour classification

    Si aucun modele entraine n'existe, utilise le classificateur
    heuristique (ImageTypeClassifier) comme fallback.
    """

    def __init__(self, n_pca_components=80, n_estimators=200,
                 model_dir: str = None):
        """
        Args:
            n_pca_components: Nombre de composantes PCA
            n_estimators: Nombre d'arbres du Random Forest
            model_dir: Repertoire de sauvegarde du modele
        """
        self.n_pca_components = n_pca_components
        self.n_estimators = n_estimators
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

        self.feature_extractor = FeatureDescriptorExtractor()
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components)
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        self._is_trained = False
        self._heuristic_fallback = ImageTypeClassifier()
        self._label_map = {}           # int -> DiagramType string
        self._inverse_label_map = {}   # DiagramType string -> int
        self._feature_names = []

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def _extract_feature_array(self, img_gray: np.ndarray,
                                color_features: Dict = None) -> np.ndarray:
        """Extrait les features et retourne un vecteur numpy"""
        fv = self.feature_extractor.extract_all(
            img_gray, color_features=color_features
        )
        return fv.to_array()

    def generate_pseudo_labels(self, images_gray: List[np.ndarray]) -> List[str]:
        """
        Genere des pseudo-labels via le classificateur heuristique.
        Utilise pour l'entrainement initial sans donnees labelisees.

        Args:
            images_gray: Liste d'images grayscale

        Returns:
            Liste de DiagramType value strings
        """
        labels = []
        for img in images_gray:
            result = classify_image_type(img)
            labels.append(result.diagram_type.value)
        return labels

    def train(self, images_gray: List[np.ndarray], labels: List[str],
              color_features_list: List[Dict] = None):
        """
        Entraine le classificateur Random Forest

        Args:
            images_gray: Liste d'images grayscale
            labels: Liste de DiagramType value strings
            color_features_list: Liste optionnelle de dicts de features couleur
        """
        # Construction des maps de labels
        unique_labels = sorted(set(labels))
        self._label_map = {i: l for i, l in enumerate(unique_labels)}
        self._inverse_label_map = {l: i for i, l in self._label_map.items()}

        y = np.array([self._inverse_label_map[l] for l in labels])

        # Extraction des features
        print(f"Extraction des features pour {len(images_gray)} images...")
        X_raw = []
        for i, img in enumerate(images_gray):
            cf = color_features_list[i] if color_features_list else None
            feat = self._extract_feature_array(img, cf)
            X_raw.append(feat)

        # Recuperer les noms de features (depuis la derniere image)
        fv = self.feature_extractor.extract_all(images_gray[-1])
        self._feature_names = fv.get_feature_names()

        X_raw = np.array(X_raw)
        print(f"Matrice de features: {X_raw.shape}")

        # Normalisation
        X_scaled = self.scaler.fit_transform(X_raw)

        # PCA (borner par le nb de features et de samples)
        actual_components = min(self.n_pca_components,
                                X_scaled.shape[1], X_scaled.shape[0] - 1)
        self.pca = PCA(n_components=max(1, actual_components))
        X_pca = self.pca.fit_transform(X_scaled)
        print(f"PCA: {X_scaled.shape[1]} -> {X_pca.shape[1]} composantes "
              f"({self.pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

        # Entrainement du Random Forest
        print(f"Entrainement du Random Forest ({self.n_estimators} arbres)...")
        self.rf.fit(X_pca, y)

        # Score de cross-validation
        n_unique = len(unique_labels)
        if len(images_gray) >= 10 and n_unique >= 2:
            cv_folds = min(5, min(np.bincount(y)))
            if cv_folds >= 2:
                scores = cross_val_score(self.rf, X_pca, y, cv=cv_folds)
                print(f"Cross-validation accuracy: {scores.mean():.3f} "
                      f"(+/- {scores.std():.3f})")

        self._is_trained = True
        print("Entrainement termine.")

    def predict(self, img_gray: np.ndarray,
                color_features: Dict = None) -> Tuple[str, float]:
        """
        Predit le type de diagramme

        Args:
            img_gray: Image grayscale
            color_features: Features couleur optionnelles

        Returns:
            (diagram_type_string, confidence)
        """
        if not self._is_trained:
            # Fallback sur heuristique
            result = classify_image_type(img_gray)
            return result.diagram_type.value, result.confidence

        feat = self._extract_feature_array(img_gray, color_features)
        feat_scaled = self.scaler.transform(feat.reshape(1, -1))
        feat_pca = self.pca.transform(feat_scaled)

        prediction = self.rf.predict(feat_pca)[0]
        probabilities = self.rf.predict_proba(feat_pca)[0]
        confidence = float(np.max(probabilities))

        return self._label_map[prediction], confidence

    def get_feature_importance(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """
        Retourne les features les plus importantes du Random Forest

        Utile pour comprendre quels concepts CV contribuent le plus
        a la classification.

        Args:
            top_n: Nombre de features a retourner

        Returns:
            Liste de (nom_feature, importance) triee par importance
        """
        if not self._is_trained:
            return []

        # L'importance est sur les composantes PCA, pas les features originales
        # On peut approximer en projetant l'importance PCA sur les features
        pca_importance = self.rf.feature_importances_
        components = self.pca.components_

        # Importance de chaque feature originale
        original_importance = np.abs(components.T @ pca_importance)

        if len(self._feature_names) != len(original_importance):
            # Fallback: juste les indices
            names = [f'feature_{i}' for i in range(len(original_importance))]
        else:
            names = self._feature_names

        pairs = list(zip(names, original_importance))
        pairs.sort(key=lambda x: x[1], reverse=True)

        return pairs[:top_n]

    def save(self, filename: str = "diagram_classifier"):
        """Sauvegarde le modele sur disque"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_data = {
            'rf': self.rf,
            'pca': self.pca,
            'scaler': self.scaler,
            'label_map': self._label_map,
            'inverse_label_map': self._inverse_label_map,
            'feature_names': self._feature_names,
            'is_trained': self._is_trained,
        }

        path = self.model_dir / f"{filename}.joblib"
        joblib.dump(model_data, path)
        print(f"Modele sauvegarde: {path}")

    def load(self, filename: str = "diagram_classifier") -> bool:
        """
        Charge le modele depuis le disque.
        Retourne True si succes, False sinon (utilisera le fallback heuristique).
        """
        path = self.model_dir / f"{filename}.joblib"

        if not path.exists():
            print(f"Pas de modele entraine trouve ({path}). Utilisation du fallback heuristique.")
            return False

        model_data = joblib.load(path)
        self.rf = model_data['rf']
        self.pca = model_data['pca']
        self.scaler = model_data['scaler']
        self._label_map = model_data['label_map']
        self._inverse_label_map = model_data['inverse_label_map']
        self._feature_names = model_data.get('feature_names', [])
        self._is_trained = model_data['is_trained']

        print(f"Modele charge: {path}")
        return True
