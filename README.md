---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# 🎙️ Speech Emotion Recognition (SER)

> An end-to-end Machine Learning project that detects human emotions from voice recordings.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-orange?logo=scikit-learn)
![librosa](https://img.shields.io/badge/Audio-librosa-green)
![Gradio](https://img.shields.io/badge/Demo-Gradio-FF7C00?logo=gradio)
![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Live Demo

**Try it instantly — no setup required:**

👉 **[https://huggingface.co/spaces/labhanshgoyal/speech-emotion-recognition](https://huggingface.co/spaces/labhanshgoyal/speech-emotion-recognition)**

Upload a `.wav` file **or record directly from your microphone** — the model predicts the speaker's emotion in real time.

---

## 📌 Project Overview

This project builds a system that classifies the **emotion of a speaker** from a short audio clip. It uses the **RAVDESS dataset** (1,440 `.wav` files, 24 actors, 8 emotions) and extracts three audio features — **MFCC, Chroma, and Mel Spectrogram** — before training a classifier.

**6 classifiers were benchmarked** (MLP, SVM, Random Forest, Gradient Boosting) and the **MLPClassifier** was selected as the best-performing model with **67.22% accuracy** and **0.67 F1-score** across 8 emotion classes.

**Recognized Emotions:** Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised

---

## 🗺️ Workflow

```
Raw Audio (.wav)
      ↓
Feature Extraction (MFCC + Chroma + Mel)     ← librosa
      ↓
Train / Test Split (75/25)                    ← sklearn
      ↓
Benchmark 6 Classifiers                      ← sklearn
      ↓
Evaluate (Accuracy, F1-Score, Confusion Matrix)
      ↓
Save Best Model + Scaler + Label Encoder (.pkl) ← pickle
      ↓
Real-Time Prediction (File or Microphone)
      ↓
Gradio Web App → Deployed on Hugging Face Spaces
```

---

## 📁 Project Structure

```
SER Project/
│
├── dataset/                  ← RAVDESS audio files (download separately)
│   ├── Actor_01/
│   └── ...
│
├── extract_features.py       ← MFCC, Chroma, Mel extraction (+ 5 optional advanced features)
├── load_data.py              ← Dataset loading, train/test split & augmentation support
├── train_model.py            ← Multi-classifier benchmarking & model saving
├── evaluate.py               ← Accuracy, F1-score, confusion matrix
├── predict.py                ← File-based CLI prediction
├── augment.py                ← Data augmentation (noise, pitch shift, time stretch)
├── explore_audio.py          ← Audio visualisation & exploration
│
├── models/
│   ├── ser_model.pkl         ← Trained MLPClassifier
│   ├── scaler.pkl            ← StandardScaler for feature normalisation
│   └── label_encoder.pkl     ← LabelEncoder for emotion class decoding
│
├── .github/
│   └── workflows/
│       └── sync_to_hf.yml    ← GitHub Actions: auto-sync to Hugging Face Spaces
│
├── confusion_matrix.png      ← Generated after evaluation
├── waveform.png              ← Generated from explore_audio.py
│
├── app.py                    ← Gradio web app (upload or microphone)
├── upload_models.py          ← Utility: upload model files to HF Spaces
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/labhanshgoyal/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

- Download **RAVDESS** from [Kaggle](https://www.kaggle.com/datasets/uwrfkiller/the-ravdess-emotional-speech-audio)
- Extract and place the `Actor_*` folders inside the `dataset/` directory

### 4. Train the Model

```bash
python train_model.py
```

This benchmarks 6 classifiers and automatically saves the best model, scaler, and label encoder to `models/`.

### 5. Evaluate the Model

```bash
python evaluate.py
```

### 6. Predict from a File (CLI)

```bash
python predict.py path/to/audio.wav
```

### 7. Launch the Web App Locally

```bash
python app.py
# Open http://localhost:7860
```

You can **upload a `.wav` file** or **record directly from your microphone** in the browser.

---

## ☁️ Deployment

The app is deployed on **Hugging Face Spaces** with automatic CI/CD via GitHub Actions.

| Item | Details |
|---|---|
| 🔗 Live URL | [huggingface.co/spaces/labhanshgoyal/speech-emotion-recognition](https://huggingface.co/spaces/labhanshgoyal/speech-emotion-recognition) |
| 🤖 CI/CD | GitHub Actions → auto-syncs to HF Spaces on every push to `master` |
| 🧠 Runtime | Gradio SDK on HF Spaces (free tier) |

---

## 📊 Results

### Classifier Benchmarking

| Classifier | Accuracy |
|---|---|
| **MLP (300) ← Best** | **67.22%** |
| MLP (256, 128) | 65.83% |
| SVM (RBF, C=50) | 63.89% |
| SVM (RBF, C=10) | 62.22% |
| Gradient Boosting | 62.22% |
| Random Forest | 54.44% |

### Best Model Metrics

| Metric | Value |
|---|---|
| Accuracy | **67.22%** |
| Weighted F1-Score | **0.67** |
| Macro Precision | 68% |
| Macro Recall | 68% |
| Features used | MFCC (40) + Chroma (12) + Mel (128) = **180 total** |
| Model | MLPClassifier — 1 hidden layer (300 neurons) |
| Dataset | RAVDESS — 1,440 audio files, 8 emotions |
| Train/Test Split | 1,080 / 360 (75% / 25%) |

### Per-Class Performance

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Angry | 0.63 | 0.65 | 0.64 |
| Calm | 0.74 | 0.76 | 0.75 |
| Disgust | 0.64 | 0.67 | 0.66 |
| Fearful | 0.62 | 0.66 | 0.64 |
| Happy | 0.72 | 0.61 | 0.66 |
| Neutral | 0.74 | 0.65 | 0.69 |
| Sad | 0.64 | 0.60 | 0.62 |
| Surprised | 0.67 | 0.81 | 0.73 |

---

## 🔬 Features Extracted

The feature extractor (`extract_features.py`) supports 8 feature types:

| Feature | Description | Size | Used in model |
|---|---|---|---|
| **MFCC** | Mel Frequency Cepstral Coefficients — voice texture & tonal quality | 40 | ✅ |
| **Chroma** | Energy distribution across 12 pitch classes | 12 | ✅ |
| **Mel Spectrogram** | Frequency-to-perception mapping of audio | 128 | ✅ |
| Delta MFCC | Temporal dynamics of MFCC | 40 | Optional |
| ZCR | Zero Crossing Rate — signal noisiness | 1 | Optional |
| RMS Energy | Signal loudness | 1 | Optional |
| Spectral Contrast | Difference in spectral peaks and valleys | 7 | Optional |
| Tonnetz | Tonal centroid features | 6 | Optional |

**Base feature vector: 180 dimensions** (MFCC + Chroma + Mel)  
**Full feature vector: 235 dimensions** (all features enabled)

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **librosa** — audio analysis and feature extraction
- **scikit-learn** — MLP, SVM, Random Forest, Gradient Boosting, metrics
- **numpy** — data handling
- **soundfile** — audio file I/O
- **gradio** — interactive web app with upload + microphone support
- **pickle** — model serialisation
- **GitHub Actions** — CI/CD pipeline to Hugging Face Spaces

---

## 📚 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors (12M + 12F)
- 8 emotions recorded at 2 intensity levels
- Download: https://www.kaggle.com/datasets/uwrfkiller/the-ravdess-emotional-speech-audio
