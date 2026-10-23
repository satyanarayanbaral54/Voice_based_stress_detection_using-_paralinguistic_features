"""
Label generation script for RAVDESS audio dataset.
Parses speech audio (channel 01), maps emotion codes to calm/stressed binary labels,
and records actor ID for leakage-free splitting.
"""

import os
import pandas as pd

DATA_DIR = "archive"
OUTPUT_CSV = "data/labels.csv"

# RAVDESS Filename format:
# Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
# Modality: 03 = audio-only
# VocalChannel: 01 = speech, 02 = song
# Emotion:
#   01 = neutral
#   02 = calm
#   03 = happy
#   04 = sad
#   05 = angry
#   06 = fearful
#   07 = disgust
#   08 = surprised

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Binary classification task:
# calm: {neutral, calm}
# stressed: {angry, fearful}
LABEL_MAP = {
    "neutral": "calm",
    "calm": "calm",
    "angry": "stressed",
    "fearful": "stressed"
}


def label_dataset(data_dir: str = DATA_DIR, output_csv: str = OUTPUT_CSV) -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    rows = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found.")

    for actor_folder in sorted(os.listdir(data_dir)):
        actor_path = os.path.join(data_dir, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        for file in sorted(os.listdir(actor_path)):
            if not file.endswith(".wav"):
                continue

            parts = file.replace(".wav", "").split("-")
            if len(parts) < 7:
                continue

            channel_code = parts[1]
            emotion_code = parts[2]
            actor_code = int(parts[6])

            # Speech only: channel 01
            if channel_code != "01":
                continue

            emotion = EMOTION_MAP.get(emotion_code)
            if emotion in LABEL_MAP:
                rows.append({
                    "filepath": os.path.join(actor_path, file).replace("\\", "/"),
                    "actor": actor_code,
                    "emotion": emotion,
                    "label": LABEL_MAP[emotion]
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Total speech files labeled: {len(df)}")
    print("Class distribution:")
    print(df["label"].value_counts())
    print("\nFiles per Actor range:")
    print(f"  Train actors (01-16): {len(df[df['actor'] <= 16])}")
    print(f"  Val actors   (17-20): {len(df[(df['actor'] >= 17) & (df['actor'] <= 20)])}")
    print(f"  Test actors  (21-24): {len(df[df['actor'] >= 21])}")
    print(f"Labels saved to: {output_csv}")
    return df


if __name__ == "__main__":
    label_dataset()
