import pickle
import gradio as gr
from extract_features import extract_feature

with open("models/ser_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

EMOTION_LABELS = {
    "neutral": "😐 Neutral",
    "calm": "😌 Calm",
    "happy": "😄 Happy",
    "sad": "😢 Sad",
    "angry": "😠 Angry",
    "fearful": "😨 Fearful",
    "disgust": "🤢 Disgust",
    "surprised": "😲 Surprised"
}

def predict_emotion(audio_path):
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True,
                               delta_mfcc=False, zcr=False, rms=False,
                               spectral_contrast=False, tonnetz=False)
    features_scaled = scaler.transform([features])
    emotion_code = model.predict(features_scaled)[0]
    emotion = le.inverse_transform([emotion_code])[0]
    return EMOTION_LABELS[emotion]

app = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", sources=["upload", "microphone"], label="Upload or Record Audio"),
    outputs=gr.Textbox(label="Detected Emotion"),
    title="Speech Emotion Recognition",
    description="Upload a .wav audio file or record from your microphone — the model will predict the speaker's emotion."
)

app.launch()