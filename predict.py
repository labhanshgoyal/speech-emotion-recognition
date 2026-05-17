import pickle
import sys
from extract_features import extract_feature

EMOTION_EMOJI = {
    "neutral": "Neutral",
    "calm": "Calm",
    "happy": "Happy",
    "sad": "Sad",
    "angry": "Angry",
    "fearful": "Fearful",
    "disgust": "Disgust",
    "surprised": "Surprised"
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/audio.wav")
        sys.exit(1)

    audio_file = sys.argv[1]
    
    with open("models/ser_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    features = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    features_scaled = scaler.transform([features])

    emotion = model.predict(features_scaled)[0]
    print(f"\nDetected Emotion: {EMOTION_EMOJI[emotion]}")