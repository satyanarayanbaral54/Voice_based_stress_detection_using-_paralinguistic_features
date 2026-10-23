"""
Shared feature extraction module for Voice-Based Stress Detection.
Used identically by data processing, training pipelines, and deployment/web apps.

Features:
  1. MFCC: 13 coefficients over 130 frames for sequence models (130, 13)
  2. Aggregated features (32 features):
     - MFCC means (13)
     - MFCC standard deviations (13)
     - Pitch mean & std via YIN (2)
     - RMS energy mean & std (2)
     - Jitter (local) via Praat/Parselmouth (1)
     - Shimmer (local) via Praat/Parselmouth (1)
     Total = 13 + 13 + 2 + 2 + 1 + 1 = 32 features.
"""

import numpy as np
import librosa
import warnings

# Praat parselmouth for acoustic perturbation analysis (Jitter & Shimmer)
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

SAMPLE_RATE = 22050
N_MFCC = 13
MAX_LEN = 130  # Fixed number of time frames for temporal models


def load_audio(filepath_or_file, sr: int = SAMPLE_RATE):
    """
    Loads and silence-trims an audio file.
    
    Args:
        filepath_or_file: Path to audio file or file-like object.
        sr: Target sample rate (default 22050 Hz).
        
    Returns:
        y: Audio time series (1D numpy array).
        sr: Sample rate.
    """
    y, sr = librosa.load(filepath_or_file, sr=sr)
    y, _ = librosa.effects.trim(y)
    return y, sr


def get_sequence(y: np.ndarray, sr: int = SAMPLE_RATE, max_len: int = MAX_LEN, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extracts time-framed MFCC sequence padded or truncated to (max_len, n_mfcc).
    No word transcription; strictly acoustic paralinguistic properties.
    
    Args:
        y: Audio time series.
        sr: Sample rate.
        max_len: Target number of time frames (130).
        n_mfcc: Number of MFCC coefficients (13).
        
    Returns:
        mfcc_seq: Numpy array of shape (max_len, n_mfcc) = (130, 13).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  # shape: (n_mfcc, time_frames)

    # Pad or truncate along time axis
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc_padded = mfcc[:, :max_len]

    # Return shape (time_frames, n_mfcc) -> (130, 13)
    return mfcc_padded.T.astype(np.float32)


def get_jitter_shimmer(y: np.ndarray, sr: int = SAMPLE_RATE):
    """
    Calculates local jitter and shimmer.
    Uses praat-parselmouth if available; falls back to librosa cycle perturbation.
    
    Returns:
        jitter: local jitter (float)
        shimmer: local shimmer (float)
    """
    jitter = 0.0
    shimmer = 0.0

    if HAS_PARSELMOUTH:
        try:
            # Ensure float64 array for Praat Sound
            sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
            # Periodic pitch analysis
            point_process = praat_call(sound, "To PointProcess (periodic, cc)", 75.0, 500.0)
            j_val = praat_call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            s_val = praat_call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

            if not np.isnan(j_val):
                jitter = float(j_val)
            if not np.isnan(s_val):
                shimmer = float(s_val)
            return jitter, shimmer
        except Exception:
            pass

    # Fallback using librosa pitch track cycle differences
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        valid = f0[(f0 > 50) & (f0 < 500)]
        if len(valid) > 2:
            periods = 1.0 / valid
            period_diffs = np.abs(np.diff(periods))
            jitter = float(np.mean(period_diffs) / np.mean(periods))

            rms = librosa.feature.rms(y=y).squeeze()
            valid_rms = rms[rms > 0]
            if len(valid_rms) > 2:
                rms_diffs = np.abs(np.diff(valid_rms))
                shimmer = float(np.mean(rms_diffs) / np.mean(valid_rms))
    except Exception:
        pass

    return jitter, shimmer


def get_aggregated(y: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extracts 32 aggregated statistical features for baseline models:
      - MFCC means (13)
      - MFCC standard deviations (13)
      - Pitch mean & std via YIN algorithm (2)
      - RMS energy mean & std (2)
      - Jitter (local) (1)
      - Shimmer (local) (1)
    Total = 13 + 13 + 2 + 2 + 1 + 1 = 32 features.
    
    Args:
        y: Audio time series.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        
    Returns:
        features: 1D numpy array of length 32.
    """
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Pitch via YIN
    pitch_values = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    valid_pitch = pitch_values[(pitch_values > 50) & (pitch_values < 500)]
    if len(valid_pitch) > 0:
        pitch_mean = float(np.mean(valid_pitch))
        pitch_std = float(np.std(valid_pitch))
    else:
        pitch_mean = 0.0
        pitch_std = 0.0

    # 3. RMS Energy
    energy = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(energy))
    energy_std = float(np.std(energy))

    # 4. Jitter and Shimmer
    jitter, shimmer = get_jitter_shimmer(y, sr=sr)

    # Combine into 32-dim vector
    agg_features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        np.array([pitch_mean, pitch_std], dtype=np.float32),
        np.array([energy_mean, energy_std], dtype=np.float32),
        np.array([jitter, shimmer], dtype=np.float32)
    ]).astype(np.float32)

    return agg_features


# Feature column names for transparency and permutation feature importance
AGGREGATED_FEATURE_NAMES = (
    [f"mfcc_mean_{i+1}" for i in range(N_MFCC)] +
    [f"mfcc_std_{i+1}" for i in range(N_MFCC)] +
    ["pitch_mean", "pitch_std"] +
    ["energy_mean", "energy_std"] +
    ["jitter", "shimmer"]
)
