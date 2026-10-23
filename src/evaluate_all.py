"""
Model evaluation script.
Loads trained models (baseline.keras, lstm.keras, cnn.keras) and tests them
on the holdout test set (Actors 21-24).
Generates:
  - Accuracy, Precision, Recall, F1-score (classification report)
  - Confusion matrices (printed and plotted to data/test_confusion_matrices.png)
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data"
MODELS_DIR = "models"


def evaluate_models():
    # 1. Load test data
    X_agg_test = np.load(os.path.join(DATA_DIR, "X_agg_test.npy"))
    X_seq_test = np.load(os.path.join(DATA_DIR, "X_seq_test.npy"))
    X_cnn_test = np.load(os.path.join(DATA_DIR, "X_cnn_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print(f"==================================================")
    print(f"EVALUATING ON UNSEEN TEST SET (Actors 21-24, N={len(y_test)})")
    print(f"Class breakdown: Calm (0) = {(y_test == 0).sum()}, Stressed (1) = {(y_test == 1).sum()}")
    print(f"==================================================\n")

    # 2. Load models
    models_dict = {
        "Baseline (Dense)": {
            "path": os.path.join(MODELS_DIR, "baseline.keras"),
            "data": X_agg_test
        },
        "LSTM (Temporal)": {
            "path": os.path.join(MODELS_DIR, "lstm.keras"),
            "data": X_seq_test
        },
        "CNN (MaxPooling)": {
            "path": os.path.join(MODELS_DIR, "cnn.keras"),
            "data": X_cnn_test
        }
    }

    results = []
    confusion_matrices = {}

    for name, config in models_dict.items():
        if not os.path.exists(config["path"]):
            print(f"[Warning] Model file not found: {config['path']}")
            continue

        model = tf.keras.models.load_model(config["path"])
        prob_preds = model.predict(config["data"], verbose=0).flatten()
        binary_preds = (prob_preds >= 0.5).astype(int)

        acc = accuracy_score(y_test, binary_preds)
        prec_w = precision_score(y_test, binary_preds, average="weighted", zero_division=0)
        rec_w = recall_score(y_test, binary_preds, average="weighted", zero_division=0)
        f1_w = f1_score(y_test, binary_preds, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, binary_preds)
        confusion_matrices[name] = cm

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec_w,
            "Recall": rec_w,
            "F1-Score": f1_w
        })

        print(f"--- {name} Classification Report ---")
        print(classification_report(y_test, binary_preds, target_names=["calm", "stressed"], digits=4))
        print("Confusion Matrix (Rows=True, Cols=Pred):")
        print(f"                Pred Calm  Pred Stressed")
        print(f"  Actual Calm       {cm[0, 0]:<10} {cm[0, 1]}")
        print(f"  Actual Stressed   {cm[1, 0]:<10} {cm[1, 1]}\n")

    # 3. Print Comparison Table
    print("\n" + "=" * 70)
    print(f"{'MODEL':<22} | {'ACCURACY':<10} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Model']:<22} | {r['Accuracy']:<10.4f} | {r['Precision']:<10.4f} | {r['Recall']:<10.4f} | {r['F1-Score']:<10.4f}")
    print("=" * 70)

    # 4. Plot Confusion Matrices
    if confusion_matrices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        cmaps = ["Blues", "Greens", "Purples"]
        for idx, (name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt="d", ax=axes[idx], cmap=cmaps[idx],
                        xticklabels=["Calm", "Stressed"], yticklabels=["Calm", "Stressed"])
            axes[idx].set_title(f"{name}\nTest Confusion Matrix")
            axes[idx].set_xlabel("Predicted Label")
            axes[idx].set_ylabel("True Label")

        plt.tight_layout()
        plot_path = os.path.join(DATA_DIR, "test_confusion_matrices.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\nSaved confusion matrix plot to: {plot_path}")

    return results


if __name__ == "__main__":
    evaluate_models()
