import glob
import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from extract_features import extract_feature
from augment import augment_audio

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


def _extract_from_audio(audio, sr):
    """Extract features from a raw audio array (used for augmented data)."""
    # Write to a temporary in-memory buffer and extract
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "_ser_aug_temp.wav")
    sf.write(tmp_path, audio, sr)
    features = extract_feature(tmp_path)
    return features


def load_data(test_size=0.25, augment=True):
    """
    Load the RAVDESS dataset and split into train/test.
    
    If augment=True, applies data augmentation to training samples
    AFTER the split (no data leakage).
    """
    x, y = [], []
    file_paths = []  # Keep paths for augmentation later

    for file in glob.glob("dataset/Actor_*/*.wav"):
        file_name = os.path.basename(file)
        emotion_code = file_name.split("-")[2]
        emotion = emotions[emotion_code]

        if emotion not in observed_emotions:
            continue

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True,
                                  delta_mfcc=False, zcr=False, rms=False,
                                  spectral_contrast=False, tonnetz=False)
        x.append(feature)
        y.append(emotion)
        file_paths.append(file)

    # Split into train/test
    x_train, x_test, y_train, y_test, paths_train, _ = train_test_split(
        np.array(x), y, file_paths, test_size=test_size, random_state=9
    )

    # Augment training data only
    if augment:
        print(f"\nAugmenting training data ({len(x_train)} samples)...")
        x_aug, y_aug = [], []

        for i, path in enumerate(paths_train):
            try:
                audio, sr = librosa.load(path, sr=None)
                augmented_audios = augment_audio(audio, sr)

                for aug_audio in augmented_audios:
                    aug_features = _extract_from_audio(aug_audio, sr)
                    x_aug.append(aug_features)
                    y_aug.append(y_train[i])
            except Exception as e:
                # Skip files that fail augmentation
                continue

            if (i + 1) % 100 == 0:
                print(f"  Augmented {i + 1}/{len(paths_train)} files...")

        # Combine original + augmented training data
        x_train = np.vstack([x_train, np.array(x_aug)])
        y_train = list(y_train) + y_aug
        print(f"  Training data after augmentation: {len(x_train)} samples")

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    print("Loading dataset and extracting features from all files...")

    x_train, x_test, y_train, y_test = load_data(augment=True)

    print(f"\nTraining samples : {len(x_train)}")
    print(f"Testing samples  : {len(x_test)}")
    print(f"Features per file: {x_train.shape[1]}")
    print(f"Sample emotions  : {y_train[:5]}")