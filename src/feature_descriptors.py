"""
Feature Descriptors - Descripteurs visuels pour classification de diagrammes

Combine HOG, LBP et ORB pour une representation riche des images.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from skimage.feature import local_binary_pattern, hog


@dataclass
class FeatureVector:
    """Vecteur de features combine pour classification"""
    hog_features: Optional[np.ndarray] = None
    lbp_features: Dict[str, float] = field(default_factory=dict)
    orb_features: Dict[str, float] = field(default_factory=dict)
    color_features: Dict[str, float] = field(default_factory=dict)
    geometric_features: Dict[str, float] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """Concatene toutes les features en un seul vecteur numpy"""
        parts = []

        if self.hog_features is not None:
            parts.append(self.hog_features.astype(np.float64))

        for feat_dict in [self.lbp_features, self.orb_features,
                          self.color_features, self.geometric_features]:
            if feat_dict:
                values = [float(v) for v in feat_dict.values()
                          if isinstance(v, (int, float, np.integer, np.floating))]
                if values:
                    parts.append(np.array(values, dtype=np.float64))

        if parts:
            return np.concatenate(parts)
        return np.array([], dtype=np.float64)

    def get_feature_names(self) -> List[str]:
        """Retourne les noms de toutes les features dans l'ordre de to_array()"""
        names = []

        if self.hog_features is not None:
            names.extend([f'hog_{i}' for i in range(len(self.hog_features))])

        for prefix, feat_dict in [('lbp', self.lbp_features),
                                   ('orb', self.orb_features),
                                   ('color', self.color_features),
                                   ('geom', self.geometric_features)]:
            if feat_dict:
                for k, v in feat_dict.items():
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        names.append(f'{prefix}_{k}')

        return names


class FeatureDescriptorExtractor:
    """
    Extracteur de descripteurs visuels (HOG, LBP, ORB)

    Usage:
        extractor = FeatureDescriptorExtractor()
        fv = extractor.extract_all(img_gray)
        feature_array = fv.to_array()  # vecteur numpy pour classification
    """

    def __init__(self, hog_orientations=9, hog_pixels_per_cell=(16, 16),
                 hog_cells_per_block=(2, 2), hog_resize=(256, 256),
                 lbp_radius=3, lbp_n_points=24,
                 orb_n_features=500):
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.hog_resize = hog_resize
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.orb_n_features = orb_n_features

    # --- HOG (Histogram of Oriented Gradients) ---

    def compute_hog_features(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Calcule le descripteur HOG (Histogram of Oriented Gradients)

        HOG divise l'image en cellules, calcule un histogramme des orientations
        de gradient dans chaque cellule, et normalise par blocs.
        Capture la distribution spatiale des contours.

        Args:
            img_gray: Image grayscale

        Returns:
            HOG feature vector (np.ndarray)
        """
        # Resize pour vecteur de taille constante
        img_resized = cv2.resize(img_gray, self.hog_resize)

        hog_features = hog(
            img_resized,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            feature_vector=True,
            block_norm='L2-Hys'
        )

        return hog_features

    # --- LBP (Local Binary Patterns) ---

    def compute_lbp_features(self, img_gray: np.ndarray) -> Dict[str, float]:
        """
        Calcule les features LBP (Local Binary Patterns)

        LBP compare chaque pixel a ses voisins pour creer un code binaire.
        L'histogramme de ces codes caracterise la texture de l'image.

        Args:
            img_gray: Image grayscale

        Returns:
            Dict avec statistiques LBP et histogramme
        """
        lbp = local_binary_pattern(img_gray, self.lbp_n_points,
                                    self.lbp_radius, method='uniform')

        # Histogramme des valeurs LBP
        n_bins = self.lbp_n_points + 2  # methode 'uniform' produit P+2 bins
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_bins), density=True)

        # Statistiques
        lbp_mean = float(np.mean(lbp))
        lbp_std = float(np.std(lbp))

        # Entropie
        hist_nonzero = hist[hist > 0]
        lbp_entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

        features = {
            'mean': lbp_mean,
            'std': lbp_std,
            'entropy': lbp_entropy,
        }

        # Bins de l'histogramme comme features
        for i, val in enumerate(hist):
            features[f'hist_{i}'] = float(val)

        return features

    # --- ORB (Oriented FAST and Rotated BRIEF) ---

    def compute_orb_features(self, img_gray: np.ndarray) -> Dict[str, float]:
        """
        Calcule les features ORB (statistiques des keypoints)

        ORB detecte des points d'interet (coins, blobs) et calcule
        un descripteur binaire pour chacun. On extrait des statistiques
        globales sur la distribution des keypoints.

        Args:
            img_gray: Image grayscale

        Returns:
            Dict avec: count, spatial_spread, mean_response, mean_size
        """
        orb = cv2.ORB_create(nfeatures=self.orb_n_features)
        keypoints = orb.detect(img_gray, None)

        if len(keypoints) == 0:
            return {
                'count': 0,
                'spatial_spread': 0.0,
                'mean_response': 0.0,
                'mean_size': 0.0,
                'spread_x': 0.0,
                'spread_y': 0.0,
            }

        # Positions des keypoints
        pts = np.array([kp.pt for kp in keypoints])
        responses = np.array([kp.response for kp in keypoints])
        sizes = np.array([kp.size for kp in keypoints])

        # Dispersion spatiale normalisee
        h, w = img_gray.shape[:2]
        spread_x = float(np.std(pts[:, 0]) / w) if w > 0 else 0.0
        spread_y = float(np.std(pts[:, 1]) / h) if h > 0 else 0.0
        spatial_spread = float(np.sqrt(spread_x**2 + spread_y**2))

        return {
            'count': len(keypoints),
            'spatial_spread': spatial_spread,
            'mean_response': float(np.mean(responses)),
            'mean_size': float(np.mean(sizes)),
            'spread_x': spread_x,
            'spread_y': spread_y,
        }

    # --- Extraction combinee ---

    def extract_all(self, img_gray: np.ndarray,
                    color_features: Dict = None,
                    geometric_features: Dict = None) -> FeatureVector:
        """
        Extrait tous les descripteurs et les combine

        Args:
            img_gray: Image grayscale
            color_features: Features couleur pre-calculees (k-means LAB)
            geometric_features: Features geometriques existantes (Hough, contours)

        Returns:
            FeatureVector combine
        """
        return FeatureVector(
            hog_features=self.compute_hog_features(img_gray),
            lbp_features=self.compute_lbp_features(img_gray),
            orb_features=self.compute_orb_features(img_gray),
            color_features=color_features or {},
            geometric_features=geometric_features or {},
        )


def extract_feature_vector(img_gray: np.ndarray, **kwargs) -> FeatureVector:
    """Fonction utilitaire pour extraction de features"""
    extractor = FeatureDescriptorExtractor()
    return extractor.extract_all(img_gray, **kwargs)
