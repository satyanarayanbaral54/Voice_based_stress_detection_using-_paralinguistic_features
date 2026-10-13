# Voice-Based Stress Detection from Paralinguistic Features

A deep learning project that classifies a voice clip as **Calm** or **Stressed** using only *how* a person speaks (tone, pitch and energy), never *what* they say. The project trains and compares three neural networks and includes a web app where a user can record or upload speech and see the predicted stress level.

> **Important:** The models are trained on *acted* emotional speech recorded in a studio. They detect speech that sounds like acted anger or fear versus calm or neutral speech. This is a student project and **not a medical or diagnostic tool**.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Features Extracted](#features-extracted)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Web App](#web-app)
- [Results](#results)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Team](#team)
- [Acknowledgements](#acknowledgements)

---

## Overview

When a person is tense, their voice changes even if the words stay the same: pitch shifts, loudness varies and tone quality changes. This project teaches a computer to pick up those paralinguistic cues.

**Key points**

- Uses only acoustic features. The audio is **never transcribed**, so the words are never used.
- Compares three architectures: a Dense network, an LSTM and a CNN with pooling.
- Splits data **by actor**, so the test voices are never seen during training.
- Runs on a normal laptop (CPU only).
- Includes a Flask web app for live recording or file upload.

## Pipeline

![Working pipeline](docs/Working_Pipeline_Diagram.png)

| Stage | What happens |
|---|---|
| 1. Dataset | RAVDESS speech clips from 24 actors |
| 2. Filter and label | Neutral + calm become Calm; angry + fearful become Stressed |
| 3. Preprocess | Load mono audio at 22,050 Hz and trim silence |
| 4. Feature extraction | MFCC grid and summary statistics (see below) |
| 5. Split by actor | Train: actors 1-16, validation: 17-20, test: 21-24 |
| 6. Scale | Scalers fitted on training data only |
| 7. Models | Baseline Dense, LSTM, CNN with pooling |
| 8. Train | Adam optimizer, binary cross-entropy, early stopping, class weights |
| 9. Evaluate | Accuracy, precision, recall, F1-score, confusion matrix on unseen actors |
| 10. Deploy | Flask backend and browser frontend |

## Dataset

This project uses the **RAVDESS** speech audio dataset (Ryerson Audio-Visual Database of Emotional Speech and Song).

| Item | Value |
|---|---|
| Total speech clips | 1,440 (24 actors, 60 clips each) |
| Clips used in this project | 672 |
| Calm (neutral + calm) | 288 |
| Stressed (angry + fearful) | 384 |
| Train / validation / test | 448 / 112 / 112 clips (actors 1-16 / 17-20 / 21-24) |

Filename format: `03-01-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav`

| Emotion code | Emotion | Project label |
|---|---|---|
| 01 | Neutral | Calm |
| 02 | Calm | Calm |
| 05 | Angry | Stressed |
| 06 | Fearful | Stressed |

The dataset is **not included** in this repository. Download it from [Zenodo](https://zenodo.org/record/1188976) or Kaggle, then place the actor folders in an `archive/` folder at the project root:

```
archive/
├── Actor_01/
├── Actor_02/
└── ... Actor_24/
```

## Features Extracted

The same code (`src/features.py`) is used for training and for the web app, so a clip is prepared identically in both places.

| Feature set | Shape | Used by |
|---|---|---|
| MFCC sequence (13 coefficients, 130 time frames) | 130 x 13 | LSTM, and CNN as a 130 x 13 x 1 "image" |
| Aggregated summary: MFCC mean and std (26), pitch mean and std (2), energy mean and std (2) | 30 values | Baseline Dense network |

Pitch is estimated with librosa's `yin` method. All features are scaled using statistics from the training set only.

## Models

| Model | Input | Architecture |
|---|---|---|
| **Baseline** | 30 summary values | Dense (64) - Dropout - Dense (32) - Dropout - Sigmoid output |
| **LSTM** | 130 x 13 sequence | LSTM (64) - Dropout - LSTM (32) - Dropout - Dense (16) - Sigmoid output |
| **CNN** | 130 x 13 x 1 image | Conv2D (16) - MaxPool - Conv2D (32) - MaxPool - Flatten - Dense (32) - Dropout - Sigmoid output |

- **Activation functions:** ReLU in hidden layers, Sigmoid at the output (a probability from 0 to 1, where 0.5 or above means Stressed).
- **Learning:** Adam optimizer (a form of gradient descent) with binary cross-entropy loss.
- **Regularisation:** dropout, early stopping and class weights for the Calm/Stressed imbalance.

## Project Structure

```
voice_stress_project/
├── archive/                 # RAVDESS dataset (not in repo)
├── src/
│   ├── features.py          # shared feature extraction
│   ├── utils.py             # training helper and plots
│   ├── label_data.py        # label clips Calm / Stressed
│   ├── extract_features.py  # MFCC + summary features
│   ├── prepare_split.py     # actor-based split and scaling
│   ├── train_baseline.py
│   ├── train_lstm.py
│   ├── train_cnn.py
│   └── evaluate_all.py      # metrics and confusion matrices
├── webapp/
│   ├── app.py               # Flask backend
│   └── templates/
│       └── index.html       # frontend
├── models/                  # saved models and scalers (generated)
├── data/                    # features, plots, results (generated)
├── docs/                    # diagrams and reports
├── requirements.txt
└── README.md
```

## Installation

**Requirements:** Python 3.9 to 3.12 and a working microphone (for live recording).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

If TensorFlow fails to install, try `pip install tensorflow-cpu`.

## How to Run

Create the empty `data/` and `models/` folders if they do not exist, place the dataset in `archive/`, then run the scripts **in this order from inside `src/`**:

```bash
cd src
python label_data.py          # creates data/labels.csv
python extract_features.py    # extracts MFCC and summary features (about 5-10 min)
python prepare_split.py       # actor-based split, scaling
python train_baseline.py      # trains the Dense model
python train_lstm.py          # trains the LSTM model
python train_cnn.py           # trains the CNN model
python evaluate_all.py        # compares all three models
```

Outputs are saved in `models/` (trained models and scalers) and `data/` (training plots, confusion matrices, `model_comparison.csv`).

## Web App

```bash
cd webapp
python app.py
```

Open **http://127.0.0.1:5000** in Chrome and allow microphone access.

> Use `127.0.0.1` or `localhost`. Browsers block microphone access on other addresses unless the page uses HTTPS.

**How it works**

1. The browser records or uploads audio and converts it to WAV in JavaScript.
2. It sends the file to the Flask backend (`POST /predict`).
3. The backend extracts the same features used in training and applies the saved scalers.
4. The three models each return a score from 0 to 1.
5. The page shows the stress level (**Low / Moderate / High**), a score bar and the score from each model.

Uploaded audio is processed temporarily and deleted after the prediction.

## Results

Results on the **test set** (actors 21-24, never seen during training). Fill in from `data/model_comparison.csv` after running `evaluate_all.py`.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Baseline (Dense) | | | | |
| LSTM | | | | |
| CNN with pooling | | | | |

Training curves (loss going down, accuracy going up) are saved in `data/` and confusion matrices in `data/confusion_matrices.png`.

> The test set is small (112 clips from 4 actors), so differences of one or two percentage points between models are not meaningful.

## Limitations

- RAVDESS contains **acted** emotions in a studio. "Stressed" here means acted anger or fear, not real psychological stress.
- Only 24 actors, so the models may not generalise to other voices, accents or languages.
- Laptop microphones and noisy rooms differ from studio recordings, so live results can be lower than test-set results.
- The dataset is small, so overfitting is a risk. Dropout, early stopping and the actor-based split reduce it.
- Not a medical or diagnostic tool.

## Future Work

- Record and test on our own calm and tense voice samples
- Add data augmentation (pitch shift, time stretch, light noise) on the training set only
- Add jitter and shimmer features
- Feature-importance analysis (pitch vs energy vs MFCC)
- Evaluate on datasets with real stress, such as SUSAS or MuSE
- Multilingual testing

## Team

| Name | Responsibility |
|---|---|
| Member 1 | Dataset and labeling |
| Member 2 | Feature extraction and preprocessing |
| Member 3 | Model building, training and evaluation |
| Member 4 | Web application and demo |

## Acknowledgements

- RAVDESS dataset: Livingstone, S. R. and Russo, F. A. (2018), *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*, PLOS ONE. Licensed under CC BY-NC-SA 4.0 (non-commercial use).
- Built with TensorFlow/Keras, librosa, scikit-learn and Flask.

## License

Add a license for your own code (for example MIT) in a `LICENSE` file. Note that the RAVDESS dataset has its own non-commercial license and is not redistributed here.
