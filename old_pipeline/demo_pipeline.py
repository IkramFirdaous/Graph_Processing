"""
Demo Pipeline - Complete demonstration on real dataset images

This script demonstrates the full CV pipeline on multiple real images
and creates comprehensive visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import ImagePreprocessor
from src.detection import GraphPrimitiveDetector
from src.segmentation import GraphSegmentator
from src.feature_extraction import GraphFeatureExtractor
from src.enrichment import TextEnricher


def demo_single_image(img_data, caption, index, preprocessor, detector, segmentator, extractor, enricher):
    """Process a single image through the complete pipeline"""

    # Load
    img_pil = load_image_from_data(img_data)
    if img_pil is None:
        return None

    img = pil_to_numpy(img_pil)

    # Pipeline
    prep_results = preprocessor.preprocess(
        img,
        grayscale=True,
        denoise_method='bilateral',
        enhance_contrast_method='clahe'
    )

    detections = detector.detect_all(prep_results['processed'])
    binary = segmentator.segment_by_threshold(prep_results['processed'])
    labels, num_comp = segmentator.connected_components(binary)
    components = segmentator.extract_component_features(labels, num_comp)

    features = extractor.extract_all_features(
        prep_results['processed'],
        nodes=detections['nodes'],
        edges=detections['edges'],
        components=components
    )

    enrichment = enricher.generate_enrichment(features)
    augmented = enricher.augment_caption(caption, enrichment)

    return {
        'index': index,
        'original': img,
        'processed': prep_results['processed'],
        'binary': binary,
        'labels': labels,
        'num_components': num_comp,
        'detections': detections,
        'features': features,
        'enrichment': enrichment,
        'caption_original': caption,
        'caption_augmented': augmented['augmented_caption']
    }


def create_summary_visualization(results_list, save_path):
    """Create a summary visualization of multiple images"""

    n_images = len(results_list)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4*n_images))

    if n_images == 1:
        axes = axes.reshape(1, -1)

    for idx, result in enumerate(results_list):
        detector = GraphPrimitiveDetector()

        # Original
        axes[idx, 0].imshow(result['original'] if result['original'].ndim == 3 else result['original'], cmap='gray')
        axes[idx, 0].set_title(f"Image {result['index']}: Original")
        axes[idx, 0].axis('off')

        # Preprocessed
        axes[idx, 1].imshow(result['processed'], cmap='gray')
        axes[idx, 1].set_title('Preprocessed')
        axes[idx, 1].axis('off')

        # Detections
        vis = detector.visualize_detections(result['processed'], result['detections'])
        axes[idx, 2].imshow(vis)
        axes[idx, 2].set_title(f"N={len(result['detections']['nodes'])}, E={len(result['detections']['edges'])}")
        axes[idx, 2].axis('off')

        # Summary text
        axes[idx, 3].axis('off')
        summary_text = (
            f"Type: {result['enrichment']['graph_type']}\n"
            f"Complexity: {result['enrichment']['complexity']}\n"
            f"Layout: {result['enrichment']['layout']}\n\n"
            f"Nodes: {result['enrichment']['node_count']}\n"
            f"Edges: {result['enrichment']['edge_count']}\n"
            f"Density: {result['enrichment']['graph_density']:.3f}\n\n"
            f"Original:\n{result['caption_original'][:60]}..."
        )
        axes[idx, 3].text(0.05, 0.5, summary_text, fontsize=8, family='monospace',
                         verticalalignment='center')
        axes[idx, 3].set_title('Analysis')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary visualization to {save_path}")
    plt.close()


def demo_pipeline(num_images=10):
    """Run complete demo on real dataset images"""

    print("="*60)
    print("COMPLETE PIPELINE DEMO - Real Dataset")
    print("="*60)

    # Create output directory
    Path("outputs/visualizations").mkdir(parents=True, exist_ok=True)
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/4] Loading dataset...")
    try:
        df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
        print(f"Dataset loaded: {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your internet connection")
        return

    # Find columns
    image_col = None
    caption_col = None
    for col in df.columns:
        if 'image' in col.lower():
            image_col = col
        if any(kw in col.lower() for kw in ['caption', 'text', 'description']):
            caption_col = col

    print(f"Using columns: image='{image_col}', caption='{caption_col}'")

    # Initialize pipeline components
    preprocessor = ImagePreprocessor(target_size=(800, 800))
    detector = GraphPrimitiveDetector()
    segmentator = GraphSegmentator()
    extractor = GraphFeatureExtractor()
    enricher = TextEnricher()

    # Process images
    print(f"\n[2/4] Processing {min(num_images, len(df))} images through complete pipeline...")

    results_list = []
    all_features = []

    for idx in range(min(num_images, len(df))):
        try:
            print(f"\n  Processing image {idx}...")

            img_data = df[image_col].iloc[idx]
            caption = df[caption_col].iloc[idx] if caption_col else "No caption"

            result = demo_single_image(
                img_data, caption, idx,
                preprocessor, detector, segmentator, extractor, enricher
            )

            if result:
                results_list.append(result)
                all_features.append(result['features'])

                # Print summary
                print(f"    Type: {result['enrichment']['graph_type']}, "
                      f"Complexity: {result['enrichment']['complexity']}")
                print(f"    Nodes: {result['enrichment']['node_count']}, "
                      f"Edges: {result['enrichment']['edge_count']}")

        except Exception as e:
            print(f"    Error: {e}")
            continue

    print(f"\nSuccessfully processed {len(results_list)} images")

    if len(results_list) == 0:
        print("No images processed successfully. Exiting.")
        return

    # Save features
    print("\n[3/4] Saving results...")
    features_df = pd.DataFrame(all_features)
    features_df.to_csv('outputs/results/demo_features.csv', index=False)
    print(f"Saved features to outputs/results/demo_features.csv")

    # Save enriched captions
    captions_data = [{
        'image_index': r['index'],
        'original': r['caption_original'],
        'augmented': r['caption_augmented'],
        'graph_type': r['enrichment']['graph_type'],
        'complexity': r['enrichment']['complexity']
    } for r in results_list]

    captions_df = pd.DataFrame(captions_data)
    captions_df.to_csv('outputs/results/demo_captions.csv', index=False)
    print(f"Saved captions to outputs/results/demo_captions.csv")

    # Create visualizations
    print("\n[4/4] Creating visualizations...")
    create_summary_visualization(results_list, 'outputs/visualizations/pipeline_demo.png')

    # Print statistics
    print("\n" + "="*60)
    print("DEMO STATISTICS")
    print("="*60)
    print(f"\nImages processed: {len(results_list)}")
    print(f"\nAverage features:")
    print(f"  Nodes: {features_df['num_nodes'].mean():.1f}")
    print(f"  Edges: {features_df['num_edges'].mean():.1f}")
    print(f"  Graph density: {features_df['graph_density'].mean():.3f}")

    # Graph types distribution
    from collections import Counter
    types = [r['enrichment']['graph_type'] for r in results_list]
    print(f"\nGraph types:")
    for graph_type, count in Counter(types).most_common():
        print(f"  {graph_type}: {count}")

    # Show sample enrichments
    print("\n" + "="*60)
    print("SAMPLE ENRICHMENTS")
    print("="*60)

    for i in range(min(3, len(results_list))):
        print(f"\nImage {results_list[i]['index']}:")
        print(f"  Original: {results_list[i]['caption_original'][:80]}...")
        print(f"  Augmented: {results_list[i]['caption_augmented'][:100]}...")

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nOutput files:")
    print("  - outputs/visualizations/pipeline_demo.png")
    print("  - outputs/results/demo_features.csv")
    print("  - outputs/results/demo_captions.csv")

    return results_list, features_df


if __name__ == "__main__":
    import sys

    # Allow specifying number of images
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    demo_pipeline(num_images=num)
