# Pipeline Legacy - Documentation

## Vue d'ensemble

Ce dossier contient l'**ancienne approche** de traitement d'images, basée sur un pipeline **uniforme et non-adaptatif**.

**Date d'archivage**: Décembre 2025
**Raison**: Remplacée par l'approche adaptative qui s'adapte automatiquement au type d'image

---

## Différences avec la méthode adaptative actuelle

### Ancienne Approche (Legacy)
```
Image → Preprocessing → Detection uniforme → Extraction uniforme → Features génériques
```

**Caractéristiques**:
- ✗ **Traitement uniforme** de toutes les images
- ✗ Extrait toujours les mêmes features (nodes, edges, components)
- ✗ Ne prend pas en compte le type spécifique de diagramme
- ✗ Moins pertinent pour des images variées (pie charts, infographics, etc.)
- ✓ Simple et direct

**Modules**:
- `detection.py` - Détection de primitives (nodes/edges) via Hough Transform
- `segmentation.py` - Segmentation en composants connexes
- `feature_extraction.py` - Extraction de features structurelles fixes
- `analysis.py` - Analyse de graphe (densité, centralité)
- `enrichment.py` - Génération de descriptions basiques

### Nouvelle Approche Adaptative (Actuelle)
```
Image → Classification du type → Extraction spécialisée → Features adaptées au type
```

**Caractéristiques**:
- ✓ **Détection automatique** du type d'image (pie chart, network, flowchart, etc.)
- ✓ **Extraction spécialisée** selon le type détecté
- ✓ Features pertinentes pour chaque type
- ✓ Descriptions enrichies et contextualisées
- ✓ Meilleure qualité d'extraction

**Modules**:
- `image_classifier.py` - Classification automatique du type d'image
- `adaptive_extractor.py` - Extraction adaptative selon le type

---

## Pourquoi ce changement?

### Problème avec l'approche uniforme
L'ancienne approche appliquait **la même logique** à toutes les images:
- Pour un **pie chart** → elle cherchait des nodes/edges (peu pertinent)
- Pour un **network graph** → extraction correcte
- Pour une **infographic** → features non adaptées

**Résultat**: Extraction sous-optimale pour 70% des images du dataset

### Avantages de l'approche adaptative
1. **Pertinence**: Chaque type d'image a ses propres extracteurs
2. **Richesse**: Plus de features spécifiques extraites
3. **Qualité**: Descriptions plus précises et utiles
4. **Flexibilité**: Facile d'ajouter de nouveaux types

---

## Cas d'usage où l'ancienne approche pourrait être pertinente

Bien que remplacée, l'approche legacy pourrait encore être utile dans certains cas:

### 1. Dataset homogène de graphes
Si vous travaillez **uniquement** avec des network graphs classiques (nodes + edges):
- L'approche uniforme est suffisante
- Plus simple à comprendre et modifier
- Pas besoin de classification

### 2. Prototype rapide
Pour un prototype très simple:
- Moins de code à comprendre
- Pipeline linéaire facile à suivre

### 3. Comparaison / Baseline
Pour comparer les performances:
- Utiliser legacy comme baseline
- Mesurer le gain de l'approche adaptative

### 4. Apprentissage
Pour apprendre les concepts CV de base:
- Pipeline plus direct et pédagogique
- Chaque étape est clairement séparée

---

## Structure des fichiers Legacy

```
legacy/old_pipeline/
├── README_LEGACY.md (ce fichier)
├── test_pipeline.py           # Test rapide sans dataset
├── explore_dataset.py          # Exploration du dataset
├── single_image_analysis.py    # Analyse détaillée d'une image
├── batch_analysis.py           # Traitement de 100 images
├── cv_validation.py            # Validation concepts CV
├── demo_pipeline.py            # Démo complète
└── src/
    ├── detection.py            # Détection primitives (Hough, contours)
    ├── segmentation.py         # Segmentation en composants
    ├── feature_extraction.py   # Extraction features structurelles
    ├── analysis.py             # Analyse de graphe
    └── enrichment.py           # Génération descriptions
```

---

## Comment utiliser les scripts Legacy

Si vous souhaitez tester l'ancienne approche:

```bash
cd old_pipeline

# Test rapide sans dataset
python test_pipeline.py

# Analyse d'une image
python single_image_analysis.py 0

# Batch analysis
python batch_analysis.py --num 50
```

**Note**: Les scripts legacy utilisent les modules dans `old_pipeline/src/`

---

## Migration vers la nouvelle approche

Si vous utilisez actuellement la legacy pipeline, voici comment migrer:

### Avant (Legacy)
```python
from src.detection import detect_primitives
from src.segmentation import segment_graph
from src.feature_extraction import extract_features

detections = detect_primitives(img)
segmentation = segment_graph(img)
features = extract_features(img, detections, segmentation)
```

### Après (Adaptative)
```python
from src.adaptive_extractor import extract_adaptive_features

# Tout en une fonction - détection automatique du type
features = extract_adaptive_features(img_processed, img_original)

# Accès aux features
print(features.diagram_type)           # Type détecté
print(features.specific_features)      # Features adaptées au type
print(features.description_enrichment) # Description enrichie
```

---

## Concepts Computer Vision couverts (Legacy)

L'ancienne approche couvrait les concepts suivants:

| Concept | Module | Fonction |
|---------|--------|----------|
| **Preprocessing** | `preprocessing.py` | Grayscale, CLAHE, bilateral filter |
| **Edge Detection** | `detection.py` | Canny edges |
| **Shape Detection** | `detection.py` | Hough Transform (circles, lines) |
| **Segmentation** | `segmentation.py` | Connected Components, Otsu threshold |
| **Contour Analysis** | `detection.py` | `cv2.findContours()` |
| **Morphology** | `segmentation.py` | Erosion, dilation |
| **Graph Analysis** | `analysis.py` | Densité, centralité |

**Note**: Tous ces concepts sont **toujours utilisés** dans l'approche adaptative, mais appliqués de manière sélective selon le type d'image.

---

## Références

- **Documentation actuelle**: Voir `README.md` à la racine
- **Code adaptatif actuel**: `src/adaptive_extractor.py`, `src/image_classifier.py`
- **Méthodologie adaptative**: Voir `METHODOLOGY_ADAPTIVE_CV.md`

---

## Questions fréquentes

### Q: Pourquoi garder le code legacy?
**R**: Pour référence historique, apprentissage, et cas d'usage spécifiques (dataset homogène).

### Q: L'ancienne approche est-elle "mauvaise"?
**R**: Non, elle fonctionne bien pour des graphes classiques. Mais elle est moins adaptée à un dataset varié.

### Q: Puis-je combiner les deux approches?
**R**: Oui! Vous pouvez utiliser la classification adaptative pour router vers l'extraction legacy pour certains types.

### Q: Le code legacy sera-t-il maintenu?
**R**: Non, il est archivé "tel quel". L'effort de développement est concentré sur l'approche adaptative.

---

**Dernière mise à jour**: Décembre 2025
