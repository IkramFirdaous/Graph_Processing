# Extraction Adaptative de Features — Diagrammes & Infographies

**Objectif** : enrichir automatiquement les captions du dataset HuggingFace [`JasmineQiuqiu/diagrams_with_captions`](https://huggingface.co/datasets/JasmineQiuqiu/diagrams_with_captions) en analysant chaque image par Computer Vision, puis en ajoutant une description structurée au caption original.

Version : **0.2.0**

---

## Architecture

Le pipeline traite chaque image par deux chemins parallèles :

```
Image RGB originale ──────────────────────────────────────────────────────┐
  │                                                                        │
  │ [grayscale]                                                            │ [RGB]
  ▼                                                                        ▼
Preprocessing                                              Segmentation couleur
  ├─ Resize (800×800)                                        K-means (k=5) en LAB
  ├─ CLAHE (contraste adaptatif)                             → num_dominant_colors
  ├─ Bilateral filter (lissage)                              → largest_segment_ratio
  ├─ Otsu thresholding (binarisation)                        → color_balance
  └─ Morphologie: opening, closing, top-hat                  → compactness
       │
       ├──► Classification du type de diagramme
       │      1. Heuristique : Hough circles/lines, contours, edge density
       │      2. Hu moments + fitEllipse pour la forme des primitives
       │      3. Random Forest (si modèle entraîné) : HOG+LBP+ORB → PCA(80d) → RF(200 arbres)
       │         ↳ fallback automatique sur l'heuristique si pas de modèle
       │
       ├──► Descripteurs visuels (FeatureVector ~8146 dims)
       │      HOG  : 256×256, 9 orientations, cells 16×16  → 8100 dims
       │      LBP  : radius=3, 24 points, uniform          → 29 stats (hist + entropy)
       │      ORB  : 500 keypoints max                     → 6 stats (count, spread, response…)
       │
       └──► Features spécifiques au type détecté
              pie_chart   → segments, angles, distribution_entropy
              network     → node_count, edge_count, clustering, avg_degree
              flowchart   → step_count, vertical_levels, branching_factor
              bar_chart   → bar_count, height_variance, orientation
              infographic → section_count, icon_count, color_diversity
```

**Sortie finale** : `AdaptiveFeatures` (dataclass) + caption enrichi

---

## Modules

| Fichier | Rôle |
|---------|------|
| `preprocessing.py` | Resize, CLAHE, bilateral filter, Otsu, morphologie (opening/closing/top-hat) |
| `detection.py` | Détection de primitives (nœuds, arêtes, texte), classification de formes par Hu moments + `fitEllipse` |
| `image_classifier.py` | Classification du type de diagramme : Hough, connected components, edge density |
| `feature_descriptors.py` | HOG, LBP, ORB → `FeatureVector` concaténé (~8146 dims) |
| `learned_classifier.py` | Pipeline `StandardScaler → PCA(80) → RandomForest(200)`, pseudo-labeling, persistance joblib |
| `adaptive_extractor.py` | Orchestre tout : double chemin RGB/gray, fusion des features, génération de description |

---

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Dépendances clés** : `opencv-python`, `scikit-learn`, `scikit-image`, `numpy`, `pandas`, `joblib`, `tqdm`

---

## Scripts

### `demo_adaptive_extraction.py`

```bash
python demo_adaptive_extraction.py 10
```

- Extrait les features sur N images
- Entraîne le Random Forest par **pseudo-labeling** (labels générés par l'heuristique, puis RF apprend à les affiner)
- Sauvegarde le modèle dans `models/diagram_classifier.joblib`
- Génère `outputs/results/adaptive_extraction.csv` et une visualisation PNG

### `process_full_dataset.py`

```bash
python process_full_dataset.py           # traitement complet (3524 images)
python process_full_dataset.py --resume  # reprendre après interruption
python process_full_dataset.py 1000      # démarrer à l'index 1000
```

- L'`AdaptiveFeatureExtractor` est instancié **une seule fois** (pas de rechargement du modèle à chaque image)
- Sauvegarde intermédiaire toutes les 500 images (`_temp.csv`)
- Sortie finale : `outputs/results/full_dataset_adaptive.csv`

---

## Format de sortie (CSV)

| Colonne | Type | Description |
|---------|------|-------------|
| `diagram_type` | str | Type détecté (`pie_chart`, `network_graph`, `flowchart`…) |
| `type_confidence` | float | Confiance de classification (0–1) |
| `visual_complexity` | float | Densité de contours normalisée |
| `color_entropy` | float | Entropie de l'histogramme couleur |
| `text_density` | float | Proportion de pixels texte estimée |
| `spatial_layout` | str | `grid`, `radial`, ou `hierarchical` |
| `color_*` | float | 5 features K-means LAB (segment ratio, balance, compacité…) |
| `lbp_entropy` | float | Entropie du descripteur LBP |
| `orb_count` | int | Nombre de keypoints ORB détectés |
| `orb_spread` | float | Dispersion spatiale des keypoints |
| `learned_type` | str | Type prédit par le RF (si modèle chargé) |
| `learned_confidence` | float | Confiance du RF |
| `specific_*` | varies | Features propres au type (ex: `specific_node_count`) |
| `original_caption` | str | Caption du dataset original |
| `enriched_caption` | str | Caption + `[Visual Analysis: …]` |

---

## Utilisation programmatique

```python
from src.preprocessing import ImagePreprocessor
from src.adaptive_extractor import AdaptiveFeatureExtractor

preprocessor = ImagePreprocessor(target_size=(800, 800))
extractor = AdaptiveFeatureExtractor(use_learned_classifier=True)

prep = preprocessor.preprocess(img, grayscale=True,
                                enhance_contrast_method='clahe',
                                morphological=True)
features = extractor.extract(prep['processed'], img_original_rgb)

print(features.diagram_type)           # 'network_graph'
print(features.type_confidence)        # 0.85
print(features.color_features)         # {'num_dominant_colors': 4, ...}
print(features.feature_vector.to_array().shape)  # (8146,)
print(features.description_enrichment) # 'Network graph with 12 nodes...'
```
