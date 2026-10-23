"""
Actor-based train/validation/test split preparation.
Strict rules:
  - Train: Actors 01 to 16
  - Validation: Actors 17 to 20
  - Test: Actors 21 to 24
  - Scalers (StandardScaler for aggregated data, channel-wise mean/std for sequence data)
    are fitted STRICTLY on the training set only to guarantee zero data leakage.
"""

import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = "data"
MODELS_DIR = "models"


def prepare_actor_splits(data_dir: str = DATA_DIR, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)

    # 1. Load extracted raw features
    X_seq = np.load(os.path.join(data_dir, "X_seq.npy"))
    X_agg = np.load(os.path.join(data_dir, "X_agg.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"), allow_pickle=True)
    actors = np.load(os.path.join(data_dir, "actors.npy"))

    print(f"Loaded {len(y)} samples. Partitioning strictly by Actor...")

    # 2. Build boolean masks for Actor-based partitions
    train_mask = actors <= 16
    val_mask = (actors >= 17) & (actors <= 20)
    test_mask = actors >= 21

    # Encode binary labels ('calm' -> 0, 'stressed' -> 1)
    le = LabelEncoder()
    # Fit encoder on train labels to be strictly safe
    le.fit(y[train_mask])
    y_encoded = le.transform(y)

    y_train = y_encoded[train_mask]
    y_val = y_encoded[val_mask]
    y_test = y_encoded[test_mask]

    # 3. Fit Aggregated Feature Scaler on TRAIN ONLY
    scaler_agg = StandardScaler()
    X_agg_train = scaler_agg.fit_transform(X_agg[train_mask])
    X_agg_val = scaler_agg.transform(X_agg[val_mask])
    X_agg_test = scaler_agg.transform(X_agg[test_mask])

    # 4. Fit Sequence Feature Scaler (channel-wise per MFCC coefficient) on TRAIN ONLY
    # X_seq shape: (samples, time_steps, n_mfcc) -> (samples, 130, 13)
    train_seq_raw = X_seq[train_mask]
    # Compute mean and std over (samples, time_steps) for each of the 13 MFCC channels
    seq_mean = np.mean(train_seq_raw, axis=(0, 1), keepdims=True)  # shape (1, 1, 13)
    seq_std = np.std(train_seq_raw, axis=(0, 1), keepdims=True)   # shape (1, 1, 13)
    seq_std[seq_std == 0] = 1.0  # Avoid zero division

    X_seq_train = (train_seq_raw - seq_mean) / seq_std
    X_seq_val = (X_seq[val_mask] - seq_mean) / seq_std
    X_seq_test = (X_seq[test_mask] - seq_mean) / seq_std

    # 5. CNN Data (samples, time_steps, n_mfcc, channels=1)
    X_cnn_train = X_seq_train[..., np.newaxis]
    X_cnn_val = X_seq_val[..., np.newaxis]
    X_cnn_test = X_seq_test[..., np.newaxis]

    # 6. Save split datasets to data/
    np.save(os.path.join(data_dir, "X_agg_train.npy"), X_agg_train.astype(np.float32))
    np.save(os.path.join(data_dir, "X_agg_val.npy"), X_agg_val.astype(np.float32))
    np.save(os.path.join(data_dir, "X_agg_test.npy"), X_agg_test.astype(np.float32))

    np.save(os.path.join(data_dir, "X_seq_train.npy"), X_seq_train.astype(np.float32))
    np.save(os.path.join(data_dir, "X_seq_val.npy"), X_seq_val.astype(np.float32))
    np.save(os.path.join(data_dir, "X_seq_test.npy"), X_seq_test.astype(np.float32))

    np.save(os.path.join(data_dir, "X_cnn_train.npy"), X_cnn_train.astype(np.float32))
    np.save(os.path.join(data_dir, "X_cnn_val.npy"), X_cnn_val.astype(np.float32))
    np.save(os.path.join(data_dir, "X_cnn_test.npy"), X_cnn_test.astype(np.float32))

    np.save(os.path.join(data_dir, "y_train.npy"), y_train)
    np.save(os.path.join(data_dir, "y_val.npy"), y_val)
    np.save(os.path.join(data_dir, "y_test.npy"), y_test)

    # 7. Save scalers and label encoder to models/
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_agg, f)

    seq_scaler_path = os.path.join(models_dir, "seq_scaler.npz")
    np.savez(seq_scaler_path, mean=seq_mean.squeeze(), std=seq_std.squeeze())

    label_enc_path = os.path.join(models_dir, "label_encoder.pkl")
    with open(label_enc_path, "wb") as f:
        pickle.dump(le, f)

    print("\nDataset Split Summary (Zero Leakage, Split by Actor):")
    print(f"  Train set (Actors 01-16): {len(y_train)} samples "
          f"(Calm: {(y_train == 0).sum()}, Stressed: {(y_train == 1).sum()})")
    print(f"  Val set   (Actors 17-20): {len(y_val)} samples "
          f"(Calm: {(y_val == 0).sum()}, Stressed: {(y_val == 1).sum()})")
    print(f"  Test set  (Actors 21-24): {len(y_test)} samples "
          f"(Calm: {(y_test == 0).sum()}, Stressed: {(y_test == 1).sum()})")
    print(f"\nSaved artifacts:")
    print(f"  - Aggregated Scaler: {scaler_path}")
    print(f"  - Sequence Scaler:   {seq_scaler_path}")
    print(f"  - Label Encoder:     {label_enc_path}")
    print(f"  - Split arrays in {data_dir}/")


if __name__ == "__main__":
    prepare_actor_splits()
