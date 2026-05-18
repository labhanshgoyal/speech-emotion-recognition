# 🎙️ Speech Emotion Recognition (SER)

> An end-to-end Machine Learning project that detects human emotions from voice recordings.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLPClassifier-orange?logo=scikit-learn)
![librosa](https://img.shields.io/badge/Audio-librosa-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project builds a system that classifies the **emotion of a speaker** from a short audio clip. It uses the **RAVDESS dataset** (1,440 `.wav` files, 24 actors, 8 emotions) and extracts three audio features — **MFCC, Chroma, and Mel Spectrogram** — before training an **MLP (Multi-Layer Perceptron) neural network** classifier.

**Recognized Emotions:** Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised

---

## 🗺️ Workflow

```
Raw Audio (.wav)
      ↓
Feature Extraction (MFCC + Chroma + Mel)  ← librosa
      ↓
Train / Test Split                         ← sklearn
      ↓
MLPClassifier (Neural Network)             ← sklearn
      ↓
Evaluate (Accuracy, Confusion Matrix)
      ↓
Save Model (.pkl)                          ← pickle
      ↓
Real-Time Prediction (Mic / File)
      ↓
(Bonus) Gradio Web App Demo
```

---

## 📁 Project Structure

```
SER Project/
│
├── dataset/              ← RAVDESS audio files (download separately)
│   ├── Actor_01/
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── extract_features.py   ← MFCC, Chroma, Mel extraction
│   ├── data_loader.py        ← Dataset loading & train/test split
│   ├── train_model.py        ← MLP training & model saving
│   ├── evaluate.py           ← Accuracy, confusion matrix
│   └── predict.py            ← Real-time / file-based prediction
│
├── models/
│   └── ser_model.pkl         ← Trained model (generated after training)
│
├── reports/
│   └── confusion_matrix.png  ← Generated after evaluation
│
├── app.py                    ← Gradio demo web app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/speech-emotion-recognition.git
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
python src/train_model.py
```

### 5. Evaluate the Model

```bash
python src/evaluate.py
```

### 6. Predict from a File

```bash
python src/predict.py --file path/to/audio.wav
```

### 7. Live Microphone Prediction

```bash
python src/predict.py
```

### 8. Launch the Web App Demo

```bash
python app.py
# Open http://localhost:7860
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | ~72%–80% |
| Features used | MFCC (40) + Chroma (12) + Mel (128) = **180 total** |
| Model | MLPClassifier — 1 hidden layer (300 neurons) |
| Dataset | RAVDESS — 1,440 audio files, 8 emotions |

---

## 🔬 Features Extracted

| Feature | Description | Size |
|---|---|---|
| **MFCC** | Mel Frequency Cepstral Coefficients — captures voice texture & tonal quality | 40 |
| **Chroma** | Energy distribution across 12 pitch classes | 12 |
| **Mel Spectrogram** | Frequency-to-perception mapping of audio | 128 |

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **librosa** — audio analysis and feature extraction
- **scikit-learn** — MLPClassifier, train/test split, metrics
- **numpy / pandas** — data handling
- **matplotlib / seaborn** — visualisation
- **gradio** — demo web app
- **pickle** — model serialisation

---

## 📚 Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors (12M + 12F)
- 8 emotions recorded at 2 intensity levels
- Download: https://www.kaggle.com/datasets/uwrfkiller/the-ravdess-emotional-speech-audio

---

## 👨‍💻 Author

Made with ❤️ as part of a Data Science learning journey.

---

## 📄 License

This project is licensed under the MIT License.
