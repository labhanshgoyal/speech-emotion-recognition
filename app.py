import pickle
import gradio as gr
from extract_features import extract_feature

with open("models/ser_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

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
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])
    emotion = model.predict(features_scaled)[0]
    return EMOTION_LABELS[emotion]
app = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(type="filepath", label="Upload a .wav audio file"),
    outputs=gr.Textbox(label="Detected Emotion"),
    title="Speech Emotion Recognition",
    description="Upload a .wav audio file and the model will predict the speaker's emotion."
)
app.launch()