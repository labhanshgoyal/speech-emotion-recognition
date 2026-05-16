import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from extract_features import extract_feature

emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}
observed_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Load data
def load_data(test_size=0.25):
    x,y = [], []

    for file in glob.glob("dataset/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions[emotion_code]

        if emotion not in observed_emotions:
            continue

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(np.array(x), y, test_size=test_size, random_state =9)

if __name__ == "__main__":
    print("Loading dataset and extracting features from all files...")
    
    x_train, x_test, y_train, y_test = load_data()
    
    print(f"Training samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Features per file: {x_train.shape[1]}")
    print(f"Sample emotions  : {y_train[:5]}")