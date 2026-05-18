---
title: Speech Emotion Recognition
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# Speech Emotion Recognition

Upload a `.wav` audio file and the model will predict the speaker's emotion.

## Emotions supported
- 😐 Neutral | 😌 Calm | 😄 Happy | 😢 Sad
- 😠 Angry | 😨 Fearful | 🤢 Disgust | 😲 Surprised

## Model
- **Architecture:** MLPClassifier (scikit-learn)
- **Features:** MFCC (40) + Chroma (12) + Mel Spectrogram (128) = 180 features
- **Dataset:** RAVDESS (1,440 audio files, 24 actors)
- **Accuracy:** 67.22%
