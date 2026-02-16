"""
Batch Analysis - Analyze multiple images from the real dataset

Objectives:
1. Understand dataset visually (distribution, types, complexity)
2. Extract information from images (nodes, edges, spatial organization)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import preprocess_image
from src.detection import detect_primitives
from src.segmentation import segment_graph
from src.feature_extraction import extract_features
from src.enrichment import TextEnricher


def analyze_batch(num_images=100):
    """Analyze a batch of images from the dataset"""
    print("="*60)
    print("BATCH ANALYSIS - Real Dataset")
    print("="*60)

    # Create output directories
    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/visualizations/batch").mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/5] Loading dataset...")
    try:
        df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
        print(f"Dataset loaded: {len(df)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Find columns
    image_col = None
    caption_col = None
    for col in df.columns:
        if 'image' in col.lower():
            image_col = col
        if any(kw in col.lower() for kw in ['caption', 'text', 'description']):
            caption_col = col

    print(f"Columns: image='{image_col}', caption='{caption_col}'")

    # Process
    all_features = []
    all_enrichments = []
    enricher = TextEnricher()

    print(f"\n[2/5] Processing {min(num_images, len(df))} images...")
    
    for idx in range(min(num_images, len(df))):
        try:
            # Load
            img_pil = load_image_from_data(df[image_col].iloc[idx])
            if img_pil is None:
                continue
            img = pil_to_numpy(img_pil)

            # Pipeline
            processed = preprocess_image(img, target_size=(600, 600))
            detections = detect_primitives(processed)
            segmentation = segment_graph(processed)
            features = extract_features(processed, detections['nodes'], detections['edges'], segmentation['components'])
            enrichment = enricher.generate_enrichment(features)

            all_features.append(features)
            all_enrichments.append(enrichment)

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1} images...")

        except Exception as e:
            if idx < 5:
                print(f"  Error on image {idx}: {e}")
            continue

    print(f"\nSuccessfully processed {len(all_features)} images")

    # Save features
    print("\n[3/5] Saving results...")
    features_df = pd.DataFrame(all_features)
    features_df.to_csv('outputs/results/dataset_features.csv', index=False)
    print(f"Saved features to outputs/results/dataset_features.csv")

    # Save enriched captions
    captions_data = []
    for idx, enrichment in enumerate(all_enrichments):
        original_caption = df[caption_col].iloc[idx] if caption_col and idx < len(df) else "No caption"
        captions_data.append({
            'original': original_caption,
            'augmented': f"{original_caption} [Visual Analysis: {enrichment['graph_type']}, {enrichment['complexity']} | Nodes: {enrichment['node_count']}, Edges: {enrichment['edge_count']}]",
            'graph_type': enrichment['graph_type'],
            'complexity': enrichment['complexity']
        })

    captions_df = pd.DataFrame(captions_data)
    captions_df.to_csv('outputs/results/enriched_captions.csv', index=False)
    print(f"Saved captions to outputs/results/enriched_captions.csv")

    # Analyze
    print("\n[4/5] Analyzing distribution...")
    print("\nStructural Statistics:")
    print(features_df[['num_nodes', 'num_edges', 'graph_density']].describe())

    # Graph types
    from collections import Counter
    types = [e['graph_type'] for e in all_enrichments]
    complexities = [e['complexity'] for e in all_enrichments]
    layouts = [e['layout'] for e in all_enrichments]

    print("\nGraph Types:")
    for t, count in Counter(types).most_common():
        print(f"  {t}: {count}")

    print("\nComplexities:")
    for c, count in Counter(complexities).most_common():
        print(f"  {c}: {count}")

    # Create distribution plots
    print("\n[5/5] Creating visualizations...")
    create_distribution_plots(features_df, all_enrichments)

    # Generate text report
    generate_report(features_df, all_enrichments, captions_df)

    print("\nComplete!")
    return features_df


def create_distribution_plots(features_df, enrichments):
    """Create distribution plots"""
    from collections import Counter

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Nodes distribution
    axes[0, 0].hist(features_df['num_nodes'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Number of Nodes')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Node Count Distribution')
    axes[0, 0].grid(True, alpha=0.3)

    # Edges distribution
    axes[0, 1].hist(features_df['num_edges'], bins=30, edgecolor='black', color='orange')
    axes[0, 1].set_xlabel('Number of Edges')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Edge Count Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # Graph density
    axes[0, 2].hist(features_df['graph_density'], bins=30, edgecolor='black', color='green')
    axes[0, 2].set_xlabel('Graph Density')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Graph Density Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # Graph types pie chart
    types = [e['graph_type'] for e in enrichments]
    type_counts = Counter(types)
    axes[1, 0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
    axes[1, 0].set_title('Graph Types')

    # Complexity bar chart
    complexities = [e['complexity'] for e in enrichments]
    complexity_counts = Counter(complexities)
    axes[1, 1].bar(complexity_counts.keys(), complexity_counts.values(), edgecolor='black')
    axes[1, 1].set_xlabel('Complexity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Complexity Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Layout bar chart
    layouts = [e['layout'] for e in enrichments]
    layout_counts = Counter(layouts)
    axes[1, 2].bar(layout_counts.keys(), layout_counts.values(), edgecolor='black', color='purple')
    axes[1, 2].set_xlabel('Layout')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Layout Distribution')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/visualizations/distribution_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved distribution plots to outputs/visualizations/distribution_analysis.png")
    plt.close()


def generate_report(features_df, enrichments, captions_df):
    """Generate text report"""
    from collections import Counter

    with open('outputs/results/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("DATASET ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")

        f.write(f"Total images analyzed: {len(features_df)}\n\n")

        f.write("-"*60 + "\n")
        f.write("STRUCTURAL STATISTICS\n")
        f.write("-"*60 + "\n\n")
        f.write(features_df[['num_nodes', 'num_edges', 'num_components', 'graph_density']].describe().to_string())

        f.write("\n\n" + "-"*60 + "\n")
        f.write("GRAPH TYPES DISTRIBUTION\n")
        f.write("-"*60 + "\n\n")
        types = [e['graph_type'] for e in enrichments]
        for graph_type, count in Counter(types).most_common():
            pct = (count / len(types)) * 100
            f.write(f"  {graph_type:15s}: {count:3d} ({pct:5.1f}%)\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("COMPLEXITY DISTRIBUTION\n")
        f.write("-"*60 + "\n\n")
        complexities = [e['complexity'] for e in enrichments]
        for complexity, count in Counter(complexities).most_common():
            pct = (count / len(complexities)) * 100
            f.write(f"  {complexity:15s}: {count:3d} ({pct:5.1f}%)\n")

        f.write("\n" + "-"*60 + "\n")
        f.write("SAMPLE ENRICHED CAPTIONS\n")
        f.write("-"*60 + "\n\n")
        for idx in range(min(5, len(captions_df))):
            f.write(f"Sample {idx + 1}:\n")
            f.write(f"  Original: {captions_df['original'].iloc[idx][:100]}...\n")
            f.write(f"  Type: {captions_df['graph_type'].iloc[idx]}\n")
            f.write(f"  Complexity: {captions_df['complexity'].iloc[idx]}\n\n")

    print("Saved report to outputs/results/analysis_report.txt")


if __name__ == "__main__":
    analyze_batch(num_images=100)
