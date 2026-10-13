nnhyny6iy111`   """
Command-line demo script for Voice-Based Stress Detection.
Usage:
    python demo.py path/to/audio.wav
"""

import os
import sys
import pickle
import warnings
import numpy as np

# Suppress TensorFlow verbosity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import load_audio, get_aggregated, get_sequence

MODELS_DIR = os.path.join(ROOT, "models")


def infer_true_label(file_path):
    fname = os.path.basename(file_path).lower()
    if "calm" in fname or "neutral" in fname:
        return "Calm"
    if "stressed" in fname or "angry" in fname or "fear" in fname:
        return "Stressed"

    # RAVDESS filename format: 03-01-XX-...
    parts = fname.replace(".wav", "").split("-")
    if len(parts) >= 3 and parts[0] == "03" and parts[1] == "01":
        emotion = parts[2]
        if emotion in ["01", "02"]:
            return "Calm"
        elif emotion in ["05", "06"]:
            return "Stressed"

    return "Unknown"


def scale_seq(seq, seq_mean, seq_std):
    try:
        if seq_mean.shape == seq.shape:
            return ((seq - seq_mean) / (seq_std + 1e-8)).astype(np.float32)
        elif seq_mean.ndim == 1 and seq_mean.shape[0] == seq.shape[1]:
            return ((seq - seq_mean) / (seq_std + 1e-8)).astype(np.float32)
        else:
            flat = seq.flatten()
            normd = (flat - seq_mean) / (seq_std + 1e-8)
            return normd.reshape(seq.shape).astype(np.float32)
    except Exception:
        return seq.astype(np.float32)


def get_level(score):
    if score < 35.0:
        return "Low"
    elif score < 65.0:
        return "Moderate"
    return "High"


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
        print("=" * 65)
        print(" Voice-Based Stress Detection — CLI Demo")
        print("=" * 65)
        print("Usage:")
        print("    python demo.py <path_to_wav_file>")
        print("\nExample:")
        print("    python demo.py demo_samples/calm_1.wav")
        print("    python demo.py demo_samples/stressed_1.wav")
        demo_dir = os.path.join(ROOT, "demo_samples")
        if os.path.exists(demo_dir):
            samples = [f for f in os.listdir(demo_dir) if f.endswith(".wav")]
            if samples:
                print("\nAvailable demo samples in demo_samples/:")
                for s in sorted(samples):
                    print(f"  - demo_samples/{s}")
        print("=" * 65)
        sys.exit(0)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"[Error] File not found: {audio_path}")
        sys.exit(1)

    if not audio_path.lower().endswith(".wav"):
        print("[Error] Only WAV audio files (.wav) are supported.")
        sys.exit(1)

    print(f"\n[1/3] Loading audio: {os.path.basename(audio_path)} ...")
    try:
        y, sr = load_audio(audio_path)
    except Exception as exc:
        print(f"[Error] Failed to read audio: {exc}")
        sys.exit(1)

    duration = len(y) / sr
    if duration < 1.0:
        print(f"[Error] Audio too short ({duration:.2f}s). Minimum 1.0 second required.")
        sys.exit(1)

    rms_global = float(np.sqrt(np.mean(y ** 2)))
    if rms_global < 1e-4:
        print("[Error] Audio appears silent or near-silent.")
        sys.exit(1)

    print(f"[2/3] Extracting paralinguistic features (duration: {duration:.2f}s, sr: {sr}Hz) ...")
    # Aggregated features (32)
    agg = get_aggregated(y, sr).reshape(1, -1)

    # Sequence features (130, 13)
    seq = get_sequence(y, sr)

    print("[3/3] Loading models and running inference ...")
    # Load scalers
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    seq_data = np.load(os.path.join(MODELS_DIR, "seq_scaler.npz"))
    seq_mean = seq_data["mean"]
    seq_std = seq_data.get("std", seq_data.get("scale", np.ones_like(seq_mean)))

    agg_s = scaler.transform(agg).astype(np.float32)
    seq_s = scale_seq(seq, seq_mean, seq_std)[np.newaxis, ...]
    cnn_s = seq_s[..., np.newaxis]

    # Load models
    baseline_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "baseline.keras"))
    lstm_model     = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm.keras"))
    cnn_model      = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn.keras"))

    p_baseline = float(baseline_model.predict(agg_s, verbose=0).flatten()[0])
    p_lstm     = float(lstm_model.predict(seq_s, verbose=0).flatten()[0])
    p_cnn      = float(cnn_model.predict(cnn_s, verbose=0).flatten()[0])

    # Ensemble: 20% Baseline + 40% LSTM + 40% CNN
    p_ensemble = 0.2 * p_baseline + 0.4 * p_lstm + 0.4 * p_cnn
    stress_score = round(p_ensemble * 100, 1)
    stress_level = get_level(stress_score)

    true_label = infer_true_label(audio_path)

    def pred_str(p):
        return "Stressed" if p >= 0.5 else "Calm"

    print("\n" + "=" * 68)
    print(f" VOICE STRESS DETECTION RESULTS")
    print("=" * 68)
    print(f" File        : {os.path.basename(audio_path)}")
    print(f" Duration    : {duration:.2f} seconds")
    if true_label != "Unknown":
        print(f" True Label  : {true_label}")
    print("-" * 68)
    print(f" {'MODEL':<20} | {'PREDICTION':<10} | {'PROBABILITY':<12} | {'SCORE':<8}")
    print("-" * 68)

    models_info = [
        ("Baseline (Dense)", p_baseline),
        ("LSTM (Temporal)",  p_lstm),
        ("CNN (MaxPooling)", p_cnn),
    ]

    for name, prob in models_info:
        p_class = pred_str(prob)
        match_icon = ""
        if true_label != "Unknown":
            match_icon = " [OK]" if p_class.lower() == true_label.lower() else " [MISMATCH]"
        print(f" {name:<20} | {p_class:<10} | {prob * 100:>6.1f}%       | {prob * 100:>5.1f}/100{match_icon}")

    print("-" * 68)
    ens_class = pred_str(p_ensemble)
    ens_icon = ""
    if true_label != "Unknown":
        ens_icon = " [CORRECT]" if ens_class.lower() == true_label.lower() else " [INCORRECT]"
    print(f" {'Ensemble Result':<20} | {ens_class:<10} | {stress_score:>6.1f}%       | {stress_score:>5.1f}/100{ens_icon}")
    print(f" Stress Level        : {stress_level} Stress")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
