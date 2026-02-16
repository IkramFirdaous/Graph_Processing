"""
Process Full Dataset - Adaptive Extraction

Applique l'extraction adaptative à tout le dataset
Génère le même format CSV que demo_adaptive_extraction.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from src.utils import load_image_from_data, pil_to_numpy
from src.preprocessing import ImagePreprocessor
from src.adaptive_extractor import extract_adaptive_features


def process_full_dataset(batch_size=100, save_every=500, start_from=0):
    """
    Traite tout le dataset avec extraction adaptative

    Args:
        batch_size: Taille des batches pour la progression
        save_every: Sauvegarde intermédiaire tous les N images
        start_from: Index de départ (pour reprendre après interruption)
    """

    print("=" * 80)
    print("TRAITEMENT COMPLET DU DATASET - EXTRACTION ADAPTATIVE")
    print("=" * 80)

    # Create output directories
    Path("outputs/results").mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n[1/4] Chargement du dataset...")
    df = pd.read_parquet("hf://datasets/JasmineQiuqiu/diagrams_with_captions/data/train-00000-of-00001.parquet")
    print(f"Dataset chargé: {len(df)} images")

    # Find columns
    image_col = [c for c in df.columns if 'image' in c.lower()][0]
    caption_col = [c for c in df.columns if 'caption' in c.lower() or 'text' in c.lower()]
    caption_col = caption_col[0] if caption_col else None

    print(f"Colonnes: image='{image_col}', caption='{caption_col}'")

    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(800, 800))

    # Results storage
    results = []
    errors = []

    # Load existing results if resuming
    output_path = 'outputs/results/full_dataset_adaptive.csv'
    temp_path = 'outputs/results/full_dataset_adaptive_temp.csv'

    if start_from > 0 and Path(temp_path).exists():
        print(f"\n[2/4] Reprise depuis l'index {start_from}...")
        existing_df = pd.read_csv(temp_path)
        results = existing_df.to_dict('records')
        print(f"Chargé {len(results)} résultats existants")
    else:
        print(f"\n[2/4] Démarrage du traitement...")

    # Process images
    print(f"\n[3/4] Traitement de {len(df)} images (depuis index {start_from})...")
    print(f"Sauvegarde intermédiaire tous les {save_every} images")

    total_processed = start_from
    start_time = time.time()

    with tqdm(total=len(df) - start_from, initial=0, desc="Processing") as pbar:
        for idx in range(start_from, len(df)):
            try:
                # Load image
                img_pil = load_image_from_data(df[image_col].iloc[idx])
                if img_pil is None:
                    errors.append({'idx': idx, 'error': 'Failed to load image'})
                    pbar.update(1)
                    continue

                img_original = pil_to_numpy(img_pil)

                # Preprocess
                prep = preprocessor.preprocess(img_original, grayscale=True, enhance_contrast_method='clahe')
                img_processed = prep['processed']

                # ADAPTIVE EXTRACTION
                features = extract_adaptive_features(img_processed, img_original)

                # Build result dict
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

                # Add specific features with prefix
                for key, value in features.specific_features.items():
                    result_dict[f'specific_{key}'] = value

                # Add captions
                if caption_col:
                    original_caption = df[caption_col].iloc[idx]
                    augmented = f"{original_caption} [Visual Analysis: {features.description_enrichment}]"
                    result_dict['original_caption'] = original_caption
                    result_dict['enriched_caption'] = augmented

                results.append(result_dict)
                total_processed += 1

                # Intermediate save
                if total_processed % save_every == 0:
                    save_intermediate(results, temp_path)

                    # Stats
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    eta = (len(df) - total_processed) / rate if rate > 0 else 0

                    pbar.set_postfix({
                        'saved': len(results),
                        'errors': len(errors),
                        'rate': f'{rate:.1f} img/s',
                        'ETA': f'{eta/60:.1f}min'
                    })

                pbar.update(1)

            except Exception as e:
                errors.append({'idx': idx, 'error': str(e)})
                pbar.update(1)
                continue

    # Final save
    print(f"\n[4/4] Sauvegarde finale...")
    print(f"Images traitées avec succès: {len(results)} / {len(df)}")
    print(f"Erreurs: {len(errors)}")

    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Résultats sauvegardés: {output_path}")

        # Remove temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()

        # Print summary
        print_summary(results_df)

    # Save errors
    if len(errors) > 0:
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv('outputs/results/processing_errors.csv', index=False)
        print(f"\n⚠ Erreurs sauvegardées: outputs/results/processing_errors.csv")

    elapsed_total = time.time() - start_time
    print(f"\n⏱ Temps total: {elapsed_total/60:.1f} minutes")
    print(f"   Vitesse moyenne: {len(results)/elapsed_total:.1f} images/seconde")

    return results_df if len(results) > 0 else None


def save_intermediate(results, path):
    """Sauvegarde intermédiaire"""
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"\n Sauvegarde intermédiaire: {len(results)} résultats")


def print_summary(results_df):
    """Affiche un résumé des résultats"""
    print("\n" + "=" * 80)
    print("RÉSUMÉ DU TRAITEMENT")
    print("=" * 80)

    # Type distribution
    type_counts = results_df['diagram_type'].value_counts()
    print(f"\n DISTRIBUTION DES TYPES ({len(results_df)} images):")
    for dtype, count in type_counts.items():
        print(f"   {dtype:20s}: {count:5d} ({count/len(results_df)*100:5.1f}%)")

    # Average confidence
    avg_confidence = results_df['type_confidence'].mean()
    print(f"\n CONFIDENCE MOYENNE: {avg_confidence * 100:.1f}%")
    print(f"   Min: {results_df['type_confidence'].min() * 100:.1f}%")
    print(f"   Max: {results_df['type_confidence'].max() * 100:.1f}%")

    # Complexity
    print(f"\n COMPLEXITÉ VISUELLE:")
    print(f"   Moyenne: {results_df['visual_complexity'].mean():.3f}")
    print(f"   Min: {results_df['visual_complexity'].min():.3f}")
    print(f"   Max: {results_df['visual_complexity'].max():.3f}")

    # Layout distribution
    if 'spatial_layout' in results_df.columns:
        layout_counts = results_df['spatial_layout'].value_counts()
        print(f"\n DISTRIBUTION DES LAYOUTS:")
        for layout, count in layout_counts.items():
            print(f"   {layout:20s}: {count:5d} ({count/len(results_df)*100:5.1f}%)")

    # Text density
    print(f"\n TEXT DENSITY:")
    print(f"   Moyenne: {results_df['text_density'].mean():.3f}")
    print(f"   Min: {results_df['text_density'].min():.3f}")
    print(f"   Max: {results_df['text_density'].max():.3f}")


def resume_processing(last_completed_idx):
    """
    Reprend le traitement après interruption

    Args:
        last_completed_idx: Dernier index complété avec succès
    """
    print(f"\n REPRISE DU TRAITEMENT depuis l'index {last_completed_idx + 1}")
    return process_full_dataset(start_from=last_completed_idx + 1)


if __name__ == "__main__":
    import sys

    # Arguments
    start_from = 0
    if len(sys.argv) > 1:
        if sys.argv[1] == '--resume':
            # Find last index from temp file
            temp_path = 'outputs/results/full_dataset_adaptive_temp.csv'
            if Path(temp_path).exists():
                temp_df = pd.read_csv(temp_path)
                start_from = temp_df['image_idx'].max() + 1
                print(f"Reprise détectée: dernier index traité = {start_from - 1}")
            else:
                print("Aucun fichier temporaire trouvé, démarrage depuis le début")
        else:
            start_from = int(sys.argv[1])

    # Process
    results = process_full_dataset(
        batch_size=100,
        save_every=500,  # Sauvegarde tous les 500 images
        start_from=start_from
    )

    print("\n" + "=" * 80)
    print("✓ TRAITEMENT TERMINÉ")
    print("=" * 80)
    print("\nFichiers générés:")
    print("  • outputs/results/full_dataset_adaptive.csv")
    if Path('outputs/results/processing_errors.csv').exists():
        print("  • outputs/results/processing_errors.csv")
