"""
Audio data augmentation for Speech Emotion Recognition.

Applies 3 augmentations to each training sample:
  1. Add Gaussian noise   — simulates noisy environments
  2. Pitch shift           — handles speaker pitch variation
  3. Time stretch          — handles speaking rate differences

Usage:
    from augment import augment_audio
    augmented_list = augment_audio(audio_array, sample_rate)
"""

import numpy as np
import librosa


def add_noise(audio, noise_factor=0.005):
    """Add random Gaussian noise to audio signal."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise


def pitch_shift(audio, sr, n_steps=2):
    """Shift pitch up or down by n_steps semitones."""
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate=1.1):
    """Speed up or slow down audio by the given rate."""
    return librosa.effects.time_stretch(y=audio, rate=rate)


def augment_audio(audio, sr):
    """
    Generate 3 augmented versions of an audio signal.
    
    Returns:
        list of numpy arrays — augmented audio signals
    """
    augmented = []

    # 1. Add noise (gentle — preserves emotion)
    augmented.append(add_noise(audio, noise_factor=0.003))

    # 2. Pitch shift (±1 semitone — small enough to preserve emotion)
    direction = np.random.choice([-1, 1])
    augmented.append(pitch_shift(audio, sr, n_steps=direction))

    # 3. Time stretch (5% variation — subtle speed change)
    rate = np.random.choice([0.95, 1.05])
    augmented.append(time_stretch(audio, rate=rate))

    return augmented


if __name__ == "__main__":
    import soundfile as sf

    TEST_FILE = "dataset/Actor_01/03-01-06-01-02-01-01.wav"
    audio, sr = librosa.load(TEST_FILE, sr=None)

    augmented = augment_audio(audio, sr)
    print(f"Original length : {len(audio)}")
    for i, aug in enumerate(augmented):
        print(f"Augmented {i+1} len : {len(aug)}")
    print(f"\nGenerated {len(augmented)} augmented versions.")
