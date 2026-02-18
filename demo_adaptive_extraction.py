"""
Montre comment le systeme ameliore s'adapte automatiquement a differents types d'images.
Inclut : segmentation couleur, descripteurs visuels (HOG/LBP/ORB),
et entrainement du classificateur Random Forest.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import ImagePreprocessor
from src.adaptive_extractor import extract_adaptive_features, AdaptiveFeatureExtractor
from src.image_classifier import DiagramType
from src.feature_descriptors import FeatureDescriptorExtractor
from src.learned_classifier import LearnedClassifier


def demo_adaptive_extraction(num_images=10):
    """
    Demo de l'extraction adaptative sur plusieurs images

    Montre:
    1. Classification automatique du type
    2. Extraction de features specifiques
    3. Segmentation couleur (K-means LAB)
    4. Descripteurs visuels (HOG, LBP, ORB)
    5. Description enrichie adaptee
    """

    # Load dataset
    print("\nChargement du dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    image_col = [c for c in df.columns if 'image' in c.lower()][0]
    caption_col = [c for c in df.columns if 'caption' in c.lower() or 'text' in c.lower()]
    caption_col = caption_col[0] if caption_col else None

    preprocessor = ImagePreprocessor(target_size=(800, 800))

    # Create output directories
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)

    results = []

    # Process images
    for idx in range(min(num_images, len(df))):
        print(f"\n{'=' * 70}")
        print(f"Image {idx}")
        print(f"{'=' * 70}")

        # Load image
        img_pil = load_image_from_data(df[image_col].iloc[idx])
        img_original = pil_to_numpy(img_pil)

        # Preprocess (with morphological cleanup)
        prep = preprocessor.preprocess(
            img_original, grayscale=True,
            enhance_contrast_method='clahe',
            morphological=True
        )
        img_processed = prep['processed']

        #   adaptative extractions
        features = extract_adaptive_features(img_processed, img_original)

        # Display results
        print(f"\n TYPE DETECTE: {features.diagram_type.upper()}")
        print(f"   Confidence: {features.type_confidence * 100:.1f}%")

        print(f"\n FEATURES UNIVERSELLES:")
        print(f"   Visual Complexity: {features.visual_complexity:.3f}")
        print(f"   Color Entropy: {features.color_entropy:.3f}")
        print(f"   Text Density: {features.text_density:.3f}")
        print(f"   Layout: {features.spatial_layout}")

        print(f"\n FEATURES SPECIFIQUES AU TYPE:")
        for key, value in features.specific_features.items():
            print(f"   {key}: {value}")

        # NEW: Color features
        if features.color_features:
            print(f"\n SEGMENTATION COULEUR (K-means LAB):")
            print(f"   Couleurs dominantes: {features.color_features['num_dominant_colors']}")
            print(f"   Plus grand segment: {features.color_features['largest_segment_ratio']:.3f}")
            print(f"   Balance couleur: {features.color_features['color_balance']:.3f}")

        # NEW: Feature descriptors
        if features.feature_vector:
            fv = features.feature_vector
            print(f"\n DESCRIPTEURS VISUELS:")
            print(f"   HOG: vecteur de {len(fv.hog_features)} dimensions" if fv.hog_features is not None else "   HOG: N/A")
            print(f"   LBP entropie: {fv.lbp_features.get('entropy', 'N/A'):.3f}" if 'entropy' in fv.lbp_features else "   LBP: N/A")
            print(f"   ORB keypoints: {fv.orb_features.get('count', 0)}")
            print(f"   ORB dispersion: {fv.orb_features.get('spatial_spread', 0):.3f}")

        print(f"\n DESCRIPTION ENRICHIE:")
        print(f"   {features.description_enrichment}")

        # Original caption
        if caption_col:
            original_caption = df[caption_col].iloc[idx]
            print(f"\n CAPTION ORIGINAL:")
            print(f"   {str(original_caption)[:200]}...")

            # Augmented caption
            augmented = f"{original_caption} [Visual Analysis: {features.description_enrichment}]"
            print(f"\n CAPTION ENRICHI:")
            print(f"   {augmented[:250]}...")

        # Save results
        result_dict = {
            'image_idx': idx,
            'diagram_type': features.diagram_type,
            'type_confidence': features.type_confidence,
            'visual_complexity': features.visual_complexity,
            'color_entropy': features.color_entropy,
            'text_density': features.text_density,
            'spatial_layout': features.spatial_layout,
            'enrichment': features.description_enrichment
        }

        # Add specific features
        for key, value in features.specific_features.items():
            result_dict[f'specific_{key}'] = value

        # Add color features
        if features.color_features:
            for key, value in features.color_features.items():
                result_dict[f'color_{key}'] = value

        # Add descriptor stats
        if features.feature_vector:
            fv = features.feature_vector
            result_dict['hog_vector_size'] = len(fv.hog_features) if fv.hog_features is not None else 0
            result_dict['lbp_entropy'] = fv.lbp_features.get('entropy', 0)
            result_dict['orb_count'] = fv.orb_features.get('count', 0)
            result_dict['orb_spread'] = fv.orb_features.get('spatial_spread', 0)

        if caption_col:
            result_dict['original_caption'] = df[caption_col].iloc[idx]
            result_dict['enriched_caption'] = augmented

        results.append(result_dict)

    # Summary
    print("\n" + "=" * 70)
    print("RESUME DE L'ANALYSE")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    # Count types
    type_counts = results_df['diagram_type'].value_counts()
    print(f"\n DISTRIBUTION DES TYPES ({num_images} images):")
    for dtype, count in type_counts.items():
        print(f"   {dtype}: {count} ({count/num_images*100:.1f}%)")

    # Average confidence
    avg_confidence = results_df['type_confidence'].mean()
    print(f"\n CONFIDENCE MOYENNE: {avg_confidence * 100:.1f}%")

    # Complexity distribution
    print(f"\n COMPLEXITE VISUELLE:")
    print(f"   Moyenne: {results_df['visual_complexity'].mean():.3f}")
    print(f"   Min: {results_df['visual_complexity'].min():.3f}")
    print(f"   Max: {results_df['visual_complexity'].max():.3f}")

    # Color features
    if 'color_num_dominant_colors' in results_df.columns:
        print(f"\n SEGMENTATION COULEUR:")
        print(f"   Couleurs dominantes (moy): {results_df['color_num_dominant_colors'].mean():.1f}")

    # Descriptors
    if 'orb_count' in results_df.columns:
        print(f"\n DESCRIPTEURS VISUELS:")
        print(f"   ORB keypoints (moy): {results_df['orb_count'].mean():.0f}")
        print(f"   LBP entropie (moy): {results_df['lbp_entropy'].mean():.3f}")

    # Save results
    results_df.to_csv('outputs/results/adaptive_extraction.csv', index=False)
    print(f"\n Resultats sauvegardes: outputs/results/adaptive_extraction.csv")

    # Visualization
    create_visualization(df, results_df, image_col, num_images)

    return results_df


def create_visualization(df, results_df, image_col, num_images):
    """Cree une visualisation montrant l'adaptation par type"""

    print("\nCreation de la visualisation...")

    # Select up to 6 images to show
    display_count = min(6, num_images)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Plot images with their detected type
    for i in range(display_count):
        row = i // 2
        col = (i % 2) * 1

        ax = fig.add_subplot(gs[row, col])

        # Load image
        img_pil = load_image_from_data(df[image_col].iloc[i])
        img = pil_to_numpy(img_pil)

        ax.imshow(img if img.ndim == 3 else img, cmap='gray' if img.ndim == 2 else None)

        # Title with type
        result = results_df.iloc[i]
        title = f"#{i}: {result['diagram_type']}\n({result['type_confidence']*100:.0f}% conf)"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

        # Add enrichment text
        enrichment_text = result['enrichment'][:80] + "..." if len(str(result['enrichment'])) > 80 else result['enrichment']
        ax.text(0.5, -0.15, enrichment_text,
                transform=ax.transAxes,
                ha='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Statistics panel
    ax_stats = fig.add_subplot(gs[:, 2])
    ax_stats.axis('off')

    # Type distribution
    type_counts = results_df['diagram_type'].value_counts()

    stats_text = f"""
EXTRACTION ADAPTATIVE 
{'=' * 35}

TYPES DETECTES ({num_images} images):

{chr(10).join([f"  {dtype}: {count} ({count/num_images*100:.0f}%)" for dtype, count in type_counts.items()])}

METRIQUES MOYENNES:
{'=' * 35}

Confidence: {results_df['type_confidence'].mean()*100:.1f}%
Visual Complexity: {results_df['visual_complexity'].mean():.3f}
Color Entropy: {results_df['color_entropy'].mean():.3f}
Text Density: {results_df['text_density'].mean():.3f}

LAYOUTS DETECTES:
{'=' * 35}

{chr(10).join([f"  {layout}: {count}" for layout, count in results_df['spatial_layout'].value_counts().items()])}

AMELIORATIONS CV:
{'=' * 35}

  Classification: Hu moments
  Filtrage: Morphologie (open/close)
  Features: HOG + LBP + ORB
  Segmentation: K-means LAB
  Reconnaissance: Random Forest
  Descripteurs: {results_df['hog_vector_size'].iloc[0] if 'hog_vector_size' in results_df.columns else 'N/A'} dim HOG
  Keypoints: {results_df['orb_count'].mean():.0f} ORB moy
"""

    ax_stats.text(0.05, 0.95, stats_text,
                  transform=ax_stats.transAxes,
                  fontsize=9, family='monospace',
                  verticalalignment='top')

    plt.suptitle("Extraction Adaptative v2 - Pipeline CV Ameliore",
                 fontsize=14, fontweight='bold')

    plt.savefig('outputs/visualizations/adaptive_extraction_demo.png', dpi=150, bbox_inches='tight')
    print("Visualisation sauvegardee: outputs/visualizations/adaptive_extraction_demo.png")

    plt.show()


def demo_training(num_images=50):
    """
    Demonstration de l'entrainement du classificateur Random Forest

    1. Charge N images
    2. Genere des pseudo-labels via le classificateur heuristique
    3. Extrait les features (HOG + LBP + ORB + couleur)
    4. Entraine le Random Forest
    5. Sauvegarde le modele
    6. Affiche l'importance des features
    """

    print("\n" + "=" * 70)
    print("ENTRAINEMENT DU CLASSIFICATEUR RANDOM FOREST")
    print("=" * 70)

    # Load dataset
    print("\nChargement du dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    image_col = [c for c in df.columns if 'image' in c.lower()][0]

    preprocessor = ImagePreprocessor(target_size=(800, 800))

    # Process images
    print(f"\nTraitement de {num_images} images pour l'entrainement...")
    images_gray = []
    color_features_list = []

    for idx in range(min(num_images, len(df))):
        img_pil = load_image_from_data(df[image_col].iloc[idx])
        if img_pil is None:
            continue

        img_original = pil_to_numpy(img_pil)

        # Preprocess
        prep = preprocessor.preprocess(
            img_original, grayscale=True,
            enhance_contrast_method='clahe',
            morphological=True
        )
        images_gray.append(prep['processed'])

        # Color features
        img_rgb = preprocessor.resize(img_original)
        from src.adaptive_extractor import AdaptiveFeatureExtractor
        temp_extractor = AdaptiveFeatureExtractor(use_learned_classifier=False)
        cf = temp_extractor._extract_color_features(img_rgb)
        color_features_list.append(cf)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{num_images} images traitees")

    # Generate pseudo-labels
    print("\nGeneration des pseudo-labels (classificateur heuristique)...")
    classifier = LearnedClassifier()
    labels = classifier.generate_pseudo_labels(images_gray)

    # Show label distribution
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nDistribution des pseudo-labels:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"   {label}: {count}")

    # Train
    print(f"\nEntrainement sur {len(images_gray)} images...")
    classifier.train(images_gray, labels, color_features_list)

    # Feature importance
    print("\nImportance des features:")
    importance = classifier.get_feature_importance(top_n=15)
    for name, imp in importance:
        bar = "#" * int(imp * 200)
        print(f"   {name:30s}: {imp:.4f} {bar}")

    # Save model
    classifier.save()

    # Test: re-classify and show results
    print("\nTest: re-classification avec le modele entraine...")
    correct = 0
    for i in range(min(10, len(images_gray))):
        pred_type, pred_conf = classifier.predict(
            images_gray[i], color_features=color_features_list[i]
        )
        original_label = labels[i]
        match = "OK" if pred_type == original_label else "DIFF"
        if pred_type == original_label:
            correct += 1
        print(f"   Image {i}: heuristique={original_label}, "
              f"RF={pred_type} ({pred_conf*100:.0f}%) [{match}]")

    print(f"\nAccord heuristique/RF: {correct}/10")

    return classifier


def compare_with_baseline(num_images=20):
    """
    Compare l'approche adaptative v2 avec la baseline (extraction uniforme)
    """

    print("\n" + "=" * 70)
    print("COMPARAISON: Adaptative v2 vs Baseline")
    print("=" * 70)

    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    image_col = [c for c in df.columns if 'image' in c.lower()][0]

    preprocessor = ImagePreprocessor(target_size=(800, 800))

    baseline_richness = []
    adaptive_richness = []

    for idx in range(min(num_images, len(df))):
        img_pil = load_image_from_data(df[image_col].iloc[idx])
        img_original = pil_to_numpy(img_pil)

        prep = preprocessor.preprocess(
            img_original, grayscale=True,
            morphological=True
        )
        img_processed = prep['processed']

        # Adaptive extraction
        features = extract_adaptive_features(img_processed, img_original)

        # Richness = number of total features extracted
        total_features = len(features.specific_features)
        if features.color_features:
            total_features += len(features.color_features)
        if features.feature_vector:
            total_features += len(features.feature_vector.to_array())
        adaptive_richness.append(total_features)

        # Baseline = always same 3 features
        baseline_richness.append(3)

    print(f"\n RICHESSE MOYENNE DES FEATURES:")
    print(f"   Baseline (uniforme): {np.mean(baseline_richness):.1f} features")
    print(f"   Adaptative v2: {np.mean(adaptive_richness):.1f} features")
    print(f"   -> Gain: +{np.mean(adaptive_richness) - np.mean(baseline_richness):.1f} features par image")

    gain_pct = (np.mean(adaptive_richness)/np.mean(baseline_richness) - 1)*100
    print(f"\n L'approche adaptative v2 extrait {gain_pct:.0f}% plus d'information!")


if __name__ == "__main__":
    import sys

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Main demo
    results = demo_adaptive_extraction(num)

    # Training demo
    print("\n")
    demo_training(min(num * 5, 100))

    # Comparison
    print("\n")
    compare_with_baseline(num)

    print("\n" + "=" * 70)
    print("DEMONSTRATION TERMINEE")
    print("=" * 70)
    print("\nFichiers generes:")
    print("  - outputs/results/adaptive_extraction.csv")
    print("  - outputs/visualizations/adaptive_extraction_demo.png")
    print("  - models/diagram_classifier.joblib (modele RF)")
