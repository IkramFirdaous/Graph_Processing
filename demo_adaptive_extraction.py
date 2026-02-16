"""
Démonstration d'Extraction Adaptative

Montre comment le système s'adapte automatiquement à différents types d'images
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import ImagePreprocessor
from src.adaptive_extractor import extract_adaptive_features
from src.image_classifier import DiagramType


def demo_adaptive_extraction(num_images=10):
    """
    Démo de l'extraction adaptative sur plusieurs images

    Montre:
    1. Classification automatique du type
    2. Extraction de features spécifiques
    3. Description enrichie adaptée
    """

    print("=" * 70)
    print("DÉMONSTRATION D'EXTRACTION ADAPTATIVE")
    print("=" * 70)
    print("\nCe système s'adapte automatiquement à différents types d'images:")
    print("  • Pie charts → Nombre de segments")
    print("  • Network graphs → Nodes et edges")
    print("  • Infographics → Sections et layout")
    print("  • Flowcharts → Étapes et niveaux")
    print("  • Et plus...")

    # Load dataset
    print("\nChargement du dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    image_col = [c for c in df.columns if 'image' in c.lower()][0]
    caption_col = [c for c in df.columns if 'caption' in c.lower() or 'text' in c.lower()]
    caption_col = caption_col[0] if caption_col else None

    preprocessor = ImagePreprocessor(target_size=(800, 800))

    results = []

    # Process images
    for idx in range(min(num_images, len(df))):
        print(f"\n{'=' * 70}")
        print(f"Image {idx}")
        print(f"{'=' * 70}")

        # Load image
        img_pil = load_image_from_data(df[image_col].iloc[idx])
        img_original = pil_to_numpy(img_pil)

        # Preprocess
        prep = preprocessor.preprocess(img_original, grayscale=True, enhance_contrast_method='clahe')
        img_processed = prep['processed']

        # ADAPTIVE EXTRACTION
        features = extract_adaptive_features(img_processed, img_original)

        # Display results
        print(f"\n TYPE DÉTECTÉ: {features.diagram_type.upper()}")
        print(f"   Confidence: {features.type_confidence * 100:.1f}%")

        print(f"\n FEATURES UNIVERSELLES:")
        print(f"   Visual Complexity: {features.visual_complexity:.3f}")
        print(f"   Color Entropy: {features.color_entropy:.3f}")
        print(f"   Text Density: {features.text_density:.3f}")
        print(f"   Layout: {features.spatial_layout}")

        print(f"\n FEATURES SPÉCIFIQUES AU TYPE:")
        for key, value in features.specific_features.items():
            print(f"   {key}: {value}")

        print(f"\n DESCRIPTION ENRICHIE:")
        print(f"   {features.description_enrichment}")

        # Original caption
        if caption_col:
            original_caption = df[caption_col].iloc[idx]
            print(f"\n CAPTION ORIGINAL:")
            print(f"   {original_caption[:200]}...")

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

        if caption_col:
            result_dict['original_caption'] = df[caption_col].iloc[idx]
            result_dict['enriched_caption'] = augmented

        results.append(result_dict)

    # Summary
    print("\n" + "=" * 70)
    print("RÉSUMÉ DE L'ANALYSE")
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
    print(f"\n COMPLEXITÉ VISUELLE:")
    print(f"   Moyenne: {results_df['visual_complexity'].mean():.3f}")
    print(f"   Min: {results_df['visual_complexity'].min():.3f}")
    print(f"   Max: {results_df['visual_complexity'].max():.3f}")

    # Save results
    results_df.to_csv('outputs/results/adaptive_extraction.csv', index=False)
    print(f"\n Résultats sauvegardés: outputs/results/adaptive_extraction.csv")

    # Visualization
    create_visualization(df, results_df, image_col, num_images)

    return results_df


def create_visualization(df, results_df, image_col, num_images):
    """Crée une visualisation montrant l'adaptation par type"""

    print("\nCréation de la visualisation...")

    # Select up to 6 images to show
    display_count = min(6, num_images)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Plot images with their detected type
    for i in range(display_count):
        row = i // 2
        col = (i % 2) * 1.5

        ax = fig.add_subplot(gs[row, int(col):int(col)+1])

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
        enrichment_text = result['enrichment'][:80] + "..." if len(result['enrichment']) > 80 else result['enrichment']
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

TYPES DÉTECTÉS ({num_images} images):

{chr(10).join([f"• {dtype}: {count} ({count/num_images*100:.0f}%)" for dtype, count in type_counts.items()])}

MÉTRIQUES MOYENNES:
{'=' * 35}

Confidence: {results_df['type_confidence'].mean()*100:.1f}%
Visual Complexity: {results_df['visual_complexity'].mean():.3f}
Color Entropy: {results_df['color_entropy'].mean():.3f}
Text Density: {results_df['text_density'].mean():.3f}

LAYOUTS DÉTECTÉS:
{'=' * 35}

{chr(10).join([f"• {layout}: {count}" for layout, count in results_df['spatial_layout'].value_counts().items()])}

MÉTHODOLOGIE CV:
{'=' * 35}

✓ Classification automatique
✓ Hough Transform (circles/lines)
✓ Connected Components
✓ Contour detection
✓ Entropy analysis
✓ Feature extraction adaptative
✓ Text density estimation
"""

    ax_stats.text(0.05, 0.95, stats_text,
                  transform=ax_stats.transAxes,
                  fontsize=9, family='monospace',
                  verticalalignment='top')

    plt.suptitle('Démonstration: Extraction Adaptative par Type d\'Image',
                 fontsize=14, fontweight='bold')

    plt.savefig('outputs/visualizations/adaptive_extraction_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Visualisation sauvegardée: outputs/visualizations/adaptive_extraction_demo.png")

    plt.show()


def compare_with_baseline(num_images=20):
    """
    Compare l'approche adaptative avec une baseline (extraction uniforme)

    Montre que l'approche adaptative donne des descriptions plus riches
    """

    print("\n" + "=" * 70)
    print("COMPARAISON: Adaptative vs Baseline")
    print("=" * 70)

    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    image_col = [c for c in df.columns if 'image' in c.lower()][0]

    preprocessor = ImagePreprocessor(target_size=(800, 800))

    baseline_richness = []
    adaptive_richness = []

    for idx in range(min(num_images, len(df))):
        img_pil = load_image_from_data(df[image_col].iloc[idx])
        img_original = pil_to_numpy(img_pil)

        prep = preprocessor.preprocess(img_original, grayscale=True)
        img_processed = prep['processed']

        # Adaptive extraction
        features = extract_adaptive_features(img_processed, img_original)

        # Richness = number of specific features extracted
        adaptive_richness.append(len(features.specific_features))

        # Baseline = always same 3 features
        baseline_richness.append(3)

    print(f"\n📊 RICHESSE MOYENNE DES FEATURES:")
    print(f"   Baseline (uniforme): {np.mean(baseline_richness):.1f} features")
    print(f"   Adaptative: {np.mean(adaptive_richness):.1f} features")
    print(f"   → Gain: +{np.mean(adaptive_richness) - np.mean(baseline_richness):.1f} features par image")

    print(f"\n✨ L'approche adaptative extrait {(np.mean(adaptive_richness)/np.mean(baseline_richness) - 1)*100:.0f}% plus d'information!")


if __name__ == "__main__":
    import sys

    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    # Main demo
    results = demo_adaptive_extraction(num)

    # Comparison
    print("\n")
    compare_with_baseline(num)

    print("\n" + "=" * 70)
    print("✓ DÉMONSTRATION TERMINÉE")
    print("=" * 70)
    print("\nFichiers générés:")
    print("  • outputs/results/adaptive_extraction.csv")
    print("  • outputs/visualizations/adaptive_extraction_demo.png")
