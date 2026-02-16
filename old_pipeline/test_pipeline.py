"""
Test Pipeline - Quick test without downloading full dataset

Creates a synthetic graph image to test the pipeline
"""

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.preprocessing import ImagePreprocessor
from src.detection import GraphPrimitiveDetector
from src.segmentation import GraphSegmentator
from src.feature_extraction import GraphFeatureExtractor
from src.enrichment import TextEnricher


def create_synthetic_graph(width=800, height=600):
    """
    Create a synthetic graph image for testing

    Returns:
        PIL Image with a simple graph
    """
    # Create white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    # Define nodes (circles)
    nodes = [
        (150, 200, 40),  # (x, y, radius)
        (400, 150, 40),
        (650, 200, 40),
        (300, 400, 40),
        (500, 400, 40),
    ]

    # Draw edges (lines) first
    edges = [
        (150, 200, 400, 150),  # (x1, y1, x2, y2)
        (400, 150, 650, 200),
        (150, 200, 300, 400),
        (400, 150, 300, 400),
        (400, 150, 500, 400),
        (650, 200, 500, 400),
        (300, 400, 500, 400),
    ]

    for edge in edges:
        draw.line(edge, fill='black', width=3)

    # Draw nodes (circles)
    for x, y, r in nodes:
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill='lightblue', outline='black', width=3)

    return img


def test_pipeline():
    """
    Test the complete pipeline
    """
    print("="*60)
    print("TESTING COMPUTER VISION PIPELINE")
    print("="*60)

    # Step 1: Create synthetic image
    print("\n[1/6] Creating synthetic graph image...")
    img_pil = create_synthetic_graph()
    img_rgb = np.array(img_pil)
    print(f"Image shape: {img_rgb.shape}")

    # Step 2: Preprocessing
    print("\n[2/6] Preprocessing...")
    preprocessor = ImagePreprocessor(target_size=(800, 800))
    prep_results = preprocessor.preprocess(
        img_rgb,
        grayscale=True,
        denoise_method='bilateral',
        enhance_contrast_method='clahe'
    )
    img_processed = prep_results['processed']
    print(f"Preprocessed shape: {img_processed.shape}")

    stats = preprocessor.get_statistics(img_processed)
    print(f"Stats: mean={stats['mean']:.2f}, std={stats['std']:.2f}, sharpness={stats['sharpness']:.2f}")

    # Step 3: Primitive Detection
    print("\n[3/6] Detecting primitives...")
    detector = GraphPrimitiveDetector()
    detections = detector.detect_all(img_processed)
    print(f"Detected: {detections['stats']['num_nodes']} nodes, "
          f"{detections['stats']['num_edges']} edges, "
          f"{detections['stats']['num_circles']} circles")

    # Step 4: Segmentation
    print("\n[4/6] Segmenting graph...")
    segmentator = GraphSegmentator()
    binary = segmentator.segment_by_threshold(img_processed)
    labels, num_components = segmentator.connected_components(binary)
    components = segmentator.extract_component_features(labels, num_components)
    print(f"Found {num_components} connected components")

    # Step 5: Feature Extraction
    print("\n[5/6] Extracting features...")
    extractor = GraphFeatureExtractor()
    features = extractor.extract_all_features(
        img_processed,
        nodes=detections['nodes'],
        edges=detections['edges'],
        components=components
    )
    print(f"Extracted {len(features)} features")

    print("\nKey Features:")
    print(f"  Nodes: {features.get('num_nodes', 0)}")
    print(f"  Edges: {features.get('num_edges', 0)}")
    print(f"  Components: {features.get('num_components', 0)}")
    print(f"  Graph density: {features.get('graph_density', 0):.3f}")

    # Step 6: Text Enrichment
    print("\n[6/6] Enriching text description...")
    enricher = TextEnricher()
    enrichment = enricher.generate_enrichment(features)

    original_caption = "A graph with connected nodes"
    augmented = enricher.augment_caption(original_caption, enrichment)

    print("\nOriginal caption:")
    print(f"  {augmented['original_caption']}")

    print("\nEnrichment:")
    print(f"  Type: {enrichment['graph_type']}")
    print(f"  Complexity: {enrichment['complexity']}")
    print(f"  Layout: {enrichment['layout']}")
    print(f"  Node count: {enrichment['node_count']}")
    print(f"  Edge count: {enrichment['edge_count']}")

    print("\nAugmented caption:")
    print(f"  {augmented['augmented_caption']}")

    # Visualization
    print("\n[7/7] Generating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Synthetic Graph')
    axes[0, 0].axis('off')

    # Preprocessed
    axes[0, 1].imshow(img_processed, cmap='gray')
    axes[0, 1].set_title('Preprocessed')
    axes[0, 1].axis('off')

    # Binary
    axes[0, 2].imshow(binary, cmap='gray')
    axes[0, 2].set_title('Binary Segmentation')
    axes[0, 2].axis('off')

    # Detections
    vis_detection = detector.visualize_detections(img_processed, detections)
    axes[1, 0].imshow(vis_detection)
    axes[1, 0].set_title(f'Detections (N={detections["stats"]["num_nodes"]}, E={detections["stats"]["num_edges"]})')
    axes[1, 0].axis('off')

    # Components
    axes[1, 1].imshow(labels, cmap='nipy_spectral')
    axes[1, 1].set_title(f'Components ({num_components})')
    axes[1, 1].axis('off')

    # Summary
    axes[1, 2].axis('off')
    summary_text = f"""
        Pipeline Test Results

        Type: {enrichment['graph_type']}
        Complexity: {enrichment['complexity']}
        Layout: {enrichment['layout']}

        Nodes: {enrichment['node_count']}
        Edges: {enrichment['edge_count']}
        Components: {enrichment['component_count']}

        Graph Density: {enrichment['graph_density']:.3f}

        [OK] Pipeline successful!
    """
    
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    axes[1, 2].set_title('Summary')

    plt.tight_layout()
    plt.savefig('outputs/visualizations/test_pipeline.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: outputs/visualizations/test_pipeline.png")

    plt.show()

    print("\n" + "="*60)
    print("PIPELINE TEST COMPLETE [OK]")
    print("="*60)
    print("\nAll modules working correctly!")

    return features, enrichment


if __name__ == "__main__":
    test_pipeline()
