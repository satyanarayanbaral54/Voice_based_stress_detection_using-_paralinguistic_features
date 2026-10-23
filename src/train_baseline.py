"""
Baseline Dense Neural Network training script.
Model 1: Baseline Dense network on 30 aggregated features (ReLU + Sigmoid)
Optimizer: Adam, Loss: Binary Cross-Entropy
Uses class weights to address calm (288) vs stressed (384) imbalance.
Validation on Actors 17-20; saves baseline.keras.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.utils.class_weight import compute_class_weight

DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "baseline.keras")


def train_baseline(epochs: int = 100, batch_size: int = 16):
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load data
    X_train = np.load(os.path.join(DATA_DIR, "X_agg_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_agg_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

    print(f"Train samples: {X_train.shape[0]} | Feature dimension: {X_train.shape[1]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # 2. Compute class weights to address imbalance
    class_weights_arr = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_arr)}
    print("Computed Class Weights:", class_weight_dict)

    # 3. Build architecture
    # Dense (64, ReLU) -> Dropout(0.3) -> Dense (32, ReLU) -> Dropout(0.3) -> Dense (1, Sigmoid)
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],), name="agg_input"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="dropout_1"),
        layers.Dense(32, activation="relu", name="dense_2"),
        layers.Dropout(0.3, name="dropout_2"),
        layers.Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="Baseline_Dense")

    model.summary()

    # 4. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    # 6. Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluate on validation
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nBaseline Model Validation Accuracy (Actors 17-20): {val_acc:.4f} (Loss: {val_loss:.4f})")

    # 8. Save in native Keras format
    model.save(MODEL_PATH)
    print(f"Saved Baseline model to: {MODEL_PATH}")
    return model, history


if __name__ == "__main__":
    train_baseline()
