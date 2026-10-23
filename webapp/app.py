"""
Voice Stress Detection - Flask Backend
"""

import os
import sys
import json
import tempfile
import logging
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory, abort
from flask_cors import CORS
import tensorflow as tf

# make src/ importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import load_audio, get_aggregated, get_sequence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR   = os.path.join(ROOT, "data")
DEMO_DIR   = os.path.join(ROOT, "demo_samples")
SAMPLES_DIR = DEMO_DIR if os.path.exists(DEMO_DIR) else os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

log.info("Loading models and scalers ...")
baseline_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "baseline.keras"))
lstm_model     = tf.keras.models.load_model(os.path.join(MODELS_DIR, "lstm.keras"))
cnn_model      = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn.keras"))

with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

seq_data = np.load(os.path.join(MODELS_DIR, "seq_scaler.npz"))
seq_mean = seq_data["mean"]
seq_std  = seq_data.get("std", seq_data.get("scale", np.ones_like(seq_data["mean"])))

with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

log.info("All models and scalers loaded OK.")


def _scale_seq(seq):
    try:
        if seq_mean.shape == seq.shape:
            return ((seq - seq_mean) / (seq_std + 1e-8)).astype(np.float32)
        elif seq_mean.ndim == 1 and seq_mean.shape[0] == seq.shape[1]:
            return ((seq - seq_mean) / (seq_std + 1e-8)).astype(np.float32)
        else:
            flat  = seq.flatten()
            normd = (flat - seq_mean) / (seq_std + 1e-8)
            return normd.reshape(seq.shape).astype(np.float32)
    except Exception:
        return seq.astype(np.float32)


def _level(score):
    if score < 35.0:
        return "Low"
    elif score < 65.0:
        return "Moderate"
    return "High"


def infer_true_label(fname):
    name = (fname or "").lower()
    if "calm" in name or "neutral" in name:
        return "Calm"
    if "stressed" in name or "angry" in name or "fear" in name:
        return "Stressed"
    parts = name.replace(".wav", "").split("-")
    if len(parts) >= 3 and parts[0] == "03" and parts[1] == "01":
        emotion = parts[2]
        if emotion in ["01", "02"]:
            return "Calm"
        elif emotion in ["05", "06"]:
            return "Stressed"
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/data/<path:filename>")
def serve_data(filename):
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        abort(400)
    return send_from_directory(DATA_DIR, filename)


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio field in request (use multipart field 'audio')."}), 400

    audio_file = request.files["audio"]
    filename   = audio_file.filename or ""
    if not filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV files are accepted."}), 415

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    try:
        audio_file.save(tmp_path)
        tmp.close()

        try:
            y, sr = load_audio(tmp_path)
        except Exception as exc:
            return jsonify({"error": f"Could not decode audio: {exc}"}), 422

        duration = float(len(y)) / sr

        if duration < 1.0:
            return jsonify({"error": f"Audio too short ({duration:.2f}s). Minimum 1 second required."}), 422

        rms_global = float(np.sqrt(np.mean(y ** 2)))
        if rms_global < 1e-4:
            return jsonify({"error": "Audio appears to be silent. Please record with a microphone."}), 422

        agg   = get_aggregated(y, sr).reshape(1, -1)
        agg_s = scaler.transform(agg).astype(np.float32)

        seq   = get_sequence(y, sr)
        seq_s = _scale_seq(seq)[np.newaxis, ...]
        cnn_s = seq_s[..., np.newaxis]

        p_baseline = float(baseline_model.predict(agg_s, verbose=0).flatten()[0])
        p_lstm     = float(lstm_model.predict(seq_s, verbose=0).flatten()[0])
        p_cnn      = float(cnn_model.predict(cnn_s, verbose=0).flatten()[0])

        p_ensemble = 0.2 * p_baseline + 0.4 * p_lstm + 0.4 * p_cnn
        stress_score = round(p_ensemble * 100, 1)
        predicted_class = "Stressed" if p_ensemble >= 0.5 else "Calm"

        # Determine ground truth label if supplied or recognizable
        req_label = request.form.get("true_label")
        true_label = req_label if req_label and req_label != "null" else infer_true_label(filename)
        is_correct = (predicted_class.lower() == true_label.lower()) if true_label else None

        import librosa
        frame_hop = 512
        pitch_contour  = librosa.yin(y, fmin=50, fmax=500, sr=sr, frame_length=2048, hop_length=frame_hop)
        # Convert any NaNs or infinities to 0
        pitch_contour = np.nan_to_num(pitch_contour, nan=0.0, posinf=0.0, neginf=0.0)
        energy_contour = librosa.feature.rms(y=y, hop_length=frame_hop).flatten()
        energy_contour = np.nan_to_num(energy_contour, nan=0.0, posinf=0.0, neginf=0.0)

        def _ds(arr, n=200):
            if len(arr) <= n:
                return [round(float(x), 2) for x in arr]
            idx = np.linspace(0, len(arr) - 1, n, dtype=int)
            return [round(float(x), 2) for x in arr[idx]]

        return jsonify({
            "filename":        filename,
            "stress_score":    stress_score,
            "level":           _level(stress_score),
            "predicted_class": predicted_class,
            "true_label":      true_label,
            "is_correct":      is_correct,
            "duration_sec":    round(duration, 2),
            "per_model": {
                "baseline": round(p_baseline * 100, 1),
                "lstm":     round(p_lstm * 100, 1),
                "cnn":      round(p_cnn * 100, 1),
            },
            "per_model_class": {
                "baseline": "Stressed" if p_baseline >= 0.5 else "Calm",
                "lstm":     "Stressed" if p_lstm >= 0.5 else "Calm",
                "cnn":      "Stressed" if p_cnn >= 0.5 else "Calm",
            },
            "pitch_contour":   _ds(pitch_contour),
            "energy_contour":  _ds(energy_contour),
        })

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.route("/samples")
def list_samples():
    try:
        clips = sorted([f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(".wav")])
        samples_detail = []
        for c in clips:
            label = infer_true_label(c) or "Unknown"
            # Format clean title e.g. "Calm 1 (Actor 21)"
            parts = c.replace(".wav", "").split("_")
            title = f"{parts[0].capitalize()} #{parts[1]}" if len(parts) == 2 else c.replace(".wav", "").capitalize()
            samples_detail.append({
                "filename": c,
                "label": label,
                "title": title
            })
        return jsonify({"samples": clips, "samples_detail": samples_detail})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/samples/<name>")
def serve_sample(name):
    if not name.lower().endswith(".wav") or "/" in name or "\\" in name:
        abort(400)
    # Check SAMPLES_DIR first, then webapp/samples
    if os.path.isfile(os.path.join(SAMPLES_DIR, name)):
        return send_from_directory(SAMPLES_DIR, name, mimetype="audio/wav")
    webapp_samples = os.path.join(os.path.dirname(__file__), "samples")
    if os.path.isfile(os.path.join(webapp_samples, name)):
        return send_from_directory(webapp_samples, name, mimetype="audio/wav")
    abort(404)


@app.route("/api/model-comparison")
def model_comparison():
    return jsonify({
        "metrics": [
            {
                "model": "Baseline (Dense)",
                "type": "Feedforward Deep Neural Network",
                "features": "32 Aggregated (MFCCs, Pitch F0, RMS, Jitter, Shimmer)",
                "input_shape": "(1, 32)",
                "accuracy": 95.54,
                "precision": 95.54,
                "recall": 95.54,
                "f1_score": 95.53,
                "notes": "Best overall classifier on test holdout; captures global vocal tension & perturbations."
            },
            {
                "model": "LSTM (Temporal)",
                "type": "Recurrent / Bidirectional LSTM",
                "features": "Sequential MFCC frames across time",
                "input_shape": "(1, 130, 13)",
                "accuracy": 92.86,
                "precision": 92.95,
                "recall": 92.86,
                "f1_score": 92.87,
                "notes": "Captures temporal cadence, speech rhythm, and continuous prosody changes."
            },
            {
                "model": "CNN (MaxPooling)",
                "type": "2D Convolutional Neural Network",
                "features": "MFCC Time-Frequency 'Spectrogram Image'",
                "input_shape": "(1, 130, 13, 1)",
                "accuracy": 92.86,
                "precision": 93.15,
                "recall": 92.86,
                "f1_score": 92.88,
                "notes": "Learns localized acoustic formant patterns and energy distribution filters."
            }
        ],
        "confusion_matrix_url": "/data/test_confusion_matrices.png",
        "test_holdout_info": {
            "actors": "Actors 21, 22, 23, 24 (Strict holdout, unseen in training/validation)",
            "sample_count": 112,
            "calm_count": 48,
            "stressed_count": 64
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
