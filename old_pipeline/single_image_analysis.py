"""
Single Image Analysis - Analyze one image in detail

This shows all steps of the CV pipeline on a real dataset image
"""

import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import ImagePreprocessor
from src.detection import GraphPrimitiveDetector
from src.segmentation import GraphSegmentator
from src.feature_extraction import GraphFeatureExtractor
from src.enrichment import TextEnricher


def analyze_single_image(image_index=0):
    """Analyze a single image from the dataset"""
    
    print("="*60)
    print(f"ANALYZING IMAGE #{image_index}")
    print("="*60)

    # Load dataset
    print("\n[1/7] Loading dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    
    # Find columns
    image_col = [c for c in df.columns if 'image' in c.lower()][0]
    caption_col = None
    for c in df.columns:
        if any(kw in c.lower() for kw in ['caption', 'text', 'description']):
            caption_col = c
            break

    # Load image
    print(f"\n[2/7] Loading image {image_index}...")
    img_pil = load_image_from_data(df[image_col].iloc[image_index])
    img = pil_to_numpy(img_pil)
    print(f"Image size: {img.shape}")

    # Preprocess
    print("\n[3/7] Preprocessing...")
    preprocessor = ImagePreprocessor(target_size=(800, 800))
    prep_results = preprocessor.preprocess(img, grayscale=True, denoise_method='bilateral', enhance_contrast_method='clahe')
    img_processed = prep_results['processed']
    
    stats = preprocessor.get_statistics(img_processed)
    print(f"Stats: mean={stats['mean']:.1f}, sharpness={stats['sharpness']:.1f}")

    # Detect
    print("\n[4/7] Detecting primitives...")
    detector = GraphPrimitiveDetector()
    detections = detector.detect_all(img_processed)
    print(f"Found: {len(detections['nodes'])} nodes, {len(detections['edges'])} edges, {len(detections['circles'])} circles")

    # Details
    print("\nNode details (first 5):")
    for i, node in enumerate(detections['nodes'][:5]):
        print(f"  Node {i}: pos={node.center}, shape={node.shape}, size={node.size}")
    
    print("\nEdge details (first 5):")
    for i, edge in enumerate(detections['edges'][:5]):
        print(f"  Edge {i}: from {edge.start} to {edge.end}, length={edge.length:.1f}px")

    # Segment
    print("\n[5/7] Segmenting...")
    segmentator = GraphSegmentator()
    binary = segmentator.segment_by_threshold(img_processed)
    labels, num_comp = segmentator.connected_components(binary)
    components = segmentator.extract_component_features(labels, num_comp)
    print(f"Found {num_comp} connected components")

    # Extract features
    print("\n[6/7] Extracting features...")
    extractor = GraphFeatureExtractor()
    features = extractor.extract_all_features(img_processed, detections['nodes'], detections['edges'], components)
    
    print(f"\nKey features:")
    print(f"  Nodes: {features.get('num_nodes', 0)}")
    print(f"  Edges: {features.get('num_edges', 0)}")
    print(f"  Components: {features.get('num_components', 0)}")
    print(f"  Graph density: {features.get('graph_density', 0):.3f}")
    print(f"  Spatial spread: x={features.get('spatial_spread_x', 0):.1f}, y={features.get('spatial_spread_y', 0):.1f}")

    # Enrich text
    print("\n[7/7] Enriching text...")
    enricher = TextEnricher()
    enrichment = enricher.generate_enrichment(features)
    
    original_caption = df[caption_col].iloc[image_index] if caption_col else "No caption"
    augmented = enricher.augment_caption(original_caption, enrichment)

    print(f"\nORIGINAL CAPTION:")
    print(f"  {original_caption}")
    
    print(f"\nENRICHMENT:")
    print(f"  Type: {enrichment['graph_type']}")
    print(f"  Complexity: {enrichment['complexity']}")
    print(f"  Layout: {enrichment['layout']}")
    print(f"  Nodes: {enrichment['node_count']}")
    print(f"  Edges: {enrichment['edge_count']}")
    
    print(f"\nAUGMENTED CAPTION:")
    print(f"  {augmented['augmented_caption']}")

    # Visualize
    print("\n[8/7] Creating visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(img if img.ndim == 3 else img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Preprocessed
    axes[0, 1].imshow(img_processed, cmap='gray')
    axes[0, 1].set_title('Preprocessed (CLAHE)')
    axes[0, 1].axis('off')

    # Binary
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary Segmentation')
    axes[0, 2].axis('off')

    # Detections
    vis = detector.visualize_detections(img_processed, detections)
    axes[1, 0].imshow(vis)
    axes[1, 0].set_title(f'Detections (N={len(detections["nodes"])}, E={len(detections["edges"])})')
    axes[1, 0].axis('off')

    # Components
    axes[1, 1].imshow(labels, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Components ({num_comp})')
    axes[1, 1].axis('off')

    # Summary
    axes[1, 2].axis('off')
    summary = f"""
Image Analysis #{image_index}

Type: {enrichment['graph_type']}
Complexity: {enrichment['complexity']}
Layout: {enrichment['layout']}

Nodes: {enrichment['node_count']}
Edges: {enrichment['edge_count']}
Components: {enrichment['component_count']}

Graph Density: {enrichment['graph_density']:.3f}

Original Caption:
{original_caption[:80]}...
"""
    axes[1, 2].text(0.05, 0.5, summary, fontsize=9, family='monospace', verticalalignment='center')
    axes[1, 2].set_title('Summary')

    plt.tight_layout()
    plt.savefig(f'outputs/visualizations/image_{image_index}_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to outputs/visualizations/image_{image_index}_analysis.png")
    
    plt.show()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

    return features, enrichment


if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    analyze_single_image(idx)
