# Speech Emotion Recognition

A machine learning project that detects human emotion from speech audio using the RAVDESS dataset.

## Project Overview

This project builds an end-to-end Speech Emotion Recognition (SER) system:
1. **Audio Exploration** — visualize raw waveforms
2. **Feature Extraction** — MFCC (40) + Chroma (12) + Mel Spectrogram (128) = 180 features per file
3. **Data Loading** — process all 1,440 RAVDESS audio files
4. **Model Training** — MLPClassifier neural network with StandardScaler
5. **Evaluation** — confusion matrix and classification report
6. **Prediction** — predict emotion from any `.wav` file
7. **Web App** — Gradio interface for live demo

## Results

- **Dataset:** RAVDESS (1,440 audio files, 24 actors, 8 emotions)
- **Model:** MLPClassifier (300 hidden neurons, ReLU activation)
- **Accuracy:** 67.22%

## Emotions Detected

| Code | Emotion |
|------|---------|
| 01 | 😐 Neutral |
| 02 | 😌 Calm |
| 03 | 😄 Happy |
| 04 | 😢 Sad |
| 05 | 😠 Angry |
| 06 | 😨 Fearful |
| 07 | 🤢 Disgust |
| 08 | 😲 Surprised |

## Project Structure

```
SER Project/
├── explore_audio.py      # Step 1: Audio visualization
├── extract_features.py   # Step 2: Feature extraction
├── load_data.py          # Step 3: Dataset loading + train/test split
├── train_model.py        # Step 4: Train + save the model
├── evaluate.py           # Step 5: Confusion matrix + classification report
├── predict.py            # Step 6: Predict emotion from audio file
├── app.py                # Step 7: Gradio web app
├── models/               # Saved model files (not tracked in git)
├── dataset/              # RAVDESS audio files (not tracked in git)
└── requirements.txt      # Python dependencies
```

## Setup

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Download Dataset

See `dataset/DOWNLOAD_DATASET.txt` for instructions on downloading the RAVDESS dataset.

## Usage

```bash
# Train the model
python train_model.py

# Evaluate the model
python evaluate.py

# Predict emotion from an audio file
python predict.py path/to/audio.wav

# Run the Gradio web app
python app.py
```

## Tech Stack

- **Python 3.x**
- **librosa** — audio feature extraction
- **scikit-learn** — MLPClassifier, StandardScaler, train_test_split
- **soundfile** — audio file reading
- **gradio** — web interface
- **matplotlib / seaborn** — visualization
