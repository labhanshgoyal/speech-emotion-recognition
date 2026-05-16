import numpy as np
import librosa
import soundfile as sf

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(X))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))

        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))

    return result

if __name__ == "__main__":
    TEST_FILE = "dataset/Actor_01/03-01-06-01-02-01-01.wav"
    features = extract_feature(TEST_FILE, mfcc=True, chroma=True, mel=True)
    print(f"Total features extracted: {len(features)}")
    print(f"First 5 values: {features[:5]}")