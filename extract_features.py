import numpy as np
import librosa
import soundfile as sf

def extract_feature(file_name, mfcc=True, chroma=True, mel=True,
                    zcr=True, rms=True, spectral_contrast=True,
                    tonnetz=True, delta_mfcc=True):
    """
    Extract audio features from a .wav file.
    
    Features (default all enabled):
        - MFCC (40) + Delta MFCC (40) = 80
        - Chroma (12)
        - Mel Spectrogram (128)
        - Zero Crossing Rate (1)
        - RMS Energy (1)
        - Spectral Contrast (7)
        - Tonnetz (6)
    Total: 235 features
    """
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        if X.ndim > 1:
            X = X.mean(axis=1)
        sample_rate = sound_file.samplerate
        result = np.array([])

        if chroma or spectral_contrast:
            stft = np.abs(librosa.stft(X))

        # ── MFCC (40) ──
        if mfcc:
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            result = np.hstack((result, mfccs_mean))

            # ── Delta MFCC (40) — captures temporal dynamics ──
            if delta_mfcc:
                delta = librosa.feature.delta(mfccs)
                delta_mean = np.mean(delta.T, axis=0)
                result = np.hstack((result, delta_mean))

        # ── Chroma (12) ──
        if chroma:
            chroma_feat = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_feat))

        # ── Mel Spectrogram (128) ──
        if mel:
            mel_feat = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_feat))

        # ── Zero Crossing Rate (1) ──
        if zcr:
            zcr_feat = np.mean(librosa.feature.zero_crossing_rate(y=X))
            result = np.hstack((result, [zcr_feat]))

        # ── RMS Energy (1) ──
        if rms:
            rms_feat = np.mean(librosa.feature.rms(y=X))
            result = np.hstack((result, [rms_feat]))

        # ── Spectral Contrast (7) ──
        if spectral_contrast:
            sc_feat = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, sc_feat))

        # ── Tonnetz (6) ──
        if tonnetz:
            harmonic = librosa.effects.harmonic(X)
            tonnetz_feat = np.mean(librosa.feature.tonnetz(y=harmonic, sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz_feat))

    return result

if __name__ == "__main__":
    TEST_FILE = "dataset/Actor_01/03-01-06-01-02-01-01.wav"
    features = extract_feature(TEST_FILE)
    print(f"Total features extracted: {len(features)}")
    print(f"First 5 values: {features[:5]}")