"""
Dataset Exploration Script
Loads and explores the diagrams_with_captions dataset
"""

import pandas as pd
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

def load_dataset():
    """Load the parquet dataset"""
    print("Loading dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def explore_structure(df):
    """Explore dataset structure"""
    print("\n" + "="*60)
    print("DATASET STRUCTURE")
    print("="*60)
    print(f"\nNumber of samples: {len(df)}")
    print(f"\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print(f"\nFirst row sample:")
    print(df.iloc[0])

    print(f"\nMemory usage:")
    print(df.memory_usage(deep=True))

    return df

def analyze_images(df, sample_size=10):
    """Analyze image properties"""
    print("\n" + "="*60)
    print("IMAGE ANALYSIS")
    print("="*60)

    # Check if images are stored as bytes or paths
    image_col = None
    for col in df.columns:
        if 'image' in col.lower() or 'img' in col.lower():
            image_col = col
            break

    if image_col is None:
        print("No image column found. Available columns:", df.columns.tolist())
        return

    print(f"\nImage column: {image_col}")
    print(f"Image data type: {type(df[image_col].iloc[0])}")

    # Sample images
    widths = []
    heights = []
    modes = []

    print(f"\nAnalyzing {min(sample_size, len(df))} sample images...")

    for idx in range(min(sample_size, len(df))):
        try:
            img_data = df[image_col].iloc[idx]

            # Try to load as PIL Image
            if isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes']))
            elif hasattr(img_data, 'convert'):  # Already a PIL Image
                img = img_data
            else:
                print(f"  Sample {idx}: Unknown image format: {type(img_data)}")
                continue

            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)

            print(f"  Sample {idx}: {img.width}x{img.height}, mode={img.mode}")

        except Exception as e:
            print(f"  Sample {idx}: Error loading image: {e}")

    if widths:
        print(f"\nImage Statistics (from {len(widths)} samples):")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.1f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.1f}")
        print(f"  Modes: {set(modes)}")

def analyze_text(df):
    """Analyze text descriptions"""
    print("\n" + "="*60)
    print("TEXT DESCRIPTION ANALYSIS")
    print("="*60)

    # Find text/caption columns
    text_cols = [col for col in df.columns if any(keyword in col.lower()
                 for keyword in ['caption', 'text', 'description', 'label'])]

    if not text_cols:
        print("No text/caption columns found")
        print("Available columns:", df.columns.tolist())
        return

    for col in text_cols:
        print(f"\nColumn: {col}")
        print(f"  Non-null count: {df[col].notna().sum()}")
        if df[col].notna().sum() > 0:
            sample = df[col].dropna().iloc[0]
            print(f"  Sample: {sample[:200]}..." if len(str(sample)) > 200 else f"  Sample: {sample}")

            # Text length statistics
            lengths = df[col].dropna().apply(lambda x: len(str(x)))
            print(f"  Length stats: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}")

def save_sample_images(df, output_dir="data/raw", num_samples=5):
    """Save sample images for visual inspection"""
    print("\n" + "="*60)
    print("SAVING SAMPLE IMAGES")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_col = None
    for col in df.columns:
        if 'image' in col.lower() or 'img' in col.lower():
            image_col = col
            break

    if image_col is None:
        print("No image column found")
        return

    for idx in range(min(num_samples, len(df))):
        try:
            img_data = df[image_col].iloc[idx]

            if isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data))
            elif isinstance(img_data, dict) and 'bytes' in img_data:
                img = Image.open(io.BytesIO(img_data['bytes']))
            elif hasattr(img_data, 'convert'):
                img = img_data
            else:
                continue

            output_file = output_path / f"sample_{idx:03d}.png"
            img.save(output_file)
            print(f"  Saved: {output_file}")

        except Exception as e:
            print(f"  Error saving sample {idx}: {e}")

def main():
    """Main exploration function"""
    print("="*60)
    print("GRAPH IMAGE DATASET EXPLORATION")
    print("="*60)

    # Load dataset
    df = load_dataset()

    # Explore structure
    explore_structure(df)

    # Analyze images
    analyze_images(df, sample_size=20)

    # Analyze text
    analyze_text(df)

    # Save samples
    save_sample_images(df, num_samples=10)

    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60)

    return df

if __name__ == "__main__":
    df = main()
