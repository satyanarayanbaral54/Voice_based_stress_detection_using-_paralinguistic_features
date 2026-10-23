"""
Feature extraction pipeline script.
Processes labeled speech audio from data/labels.csv using src.features.
Extracts:
  - Sequence data: (130, 13) MFCC sequence
  - Aggregated data: 30 statistical features (MFCC mean/std, YIN pitch mean/std, RMS mean/std)
Saves arrays to data/
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from features import load_audio, get_sequence, get_aggregated

LABELS_CSV = "data/labels.csv"
OUTPUT_DIR = "data"


def extract_all_features(labels_csv: str = LABELS_CSV, output_dir: str = OUTPUT_DIR):
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(f"'{labels_csv}' not found. Run src/label_data.py first.")

    df = pd.read_csv(labels_csv)
    print(f"Extracting acoustic paralinguistic features from {len(df)} files...")

    sequence_data = []
    aggregated_data = []
    labels = []
    actors = []
    filepaths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        filepath = row["filepath"]
        try:
            y, sr = load_audio(filepath)
            seq = get_sequence(y, sr=sr)
            agg = get_aggregated(y, sr=sr)

            sequence_data.append(seq)
            aggregated_data.append(agg)
            labels.append(row["label"])
            actors.append(int(row["actor"]))
            filepaths.append(filepath)
        except Exception as e:
            print(f"\n[Warning] Skipped {filepath}: {e}")

    X_seq = np.array(sequence_data, dtype=np.float32)
    X_agg = np.array(aggregated_data, dtype=np.float32)
    y = np.array(labels)
    actors = np.array(actors, dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_seq.npy"), X_seq)
    np.save(os.path.join(output_dir, "X_agg.npy"), X_agg)
    np.save(os.path.join(output_dir, "y.npy"), y)
    np.save(os.path.join(output_dir, "actors.npy"), actors)

    print("\nFeature extraction summary:")
    print(f"  Sequence shape   (X_seq): {X_seq.shape}  [Time frames: 130, MFCCs: 13]")
    print(f"  Aggregated shape (X_agg): {X_agg.shape}  [Acoustic features: 30]")
    print(f"  Labels shape     (y)    : {y.shape}")
    print(f"  Actors shape     (actor): {actors.shape}")
    print(f"All features saved successfully to '{output_dir}/'.")


if __name__ == "__main__":
    extract_all_features()
