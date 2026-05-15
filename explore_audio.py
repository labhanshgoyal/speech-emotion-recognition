# ============================================================
# Step 1: Exploring Audio Data
# Goal: Load ONE audio file and understand what it looks
#        like as raw numbers. This is NOT ML yet.
# Run this file with: python explore_audio.py
# ============================================================

# What is an import?
# Python doesn't know about audio or math by default.
# "import" brings in extra tools (called libraries) that
# other people have already built, so we don't start from scratch.

import librosa            # A library made specifically for audio analysis
import numpy as np        # A library for working with arrays (lists of numbers)
import matplotlib.pyplot as plt  # A library for drawing charts and graphs

# ── What is a constant? ──────────────────────────────────────
# AUDIO_FILE is a variable written in CAPS — this is a Python
# convention meaning "this value won't change during the program".
# It stores the path (location) of the audio file we want to explore.
#
# The filename 03-01-06-01-02-01-01.wav encodes information:
#   Position 1 → 03 = audio-only recording
#   Position 2 → 01 = speech (not song)
#   Position 3 → 06 = FEARFUL (the emotion) ← most important!
#   Position 7 → 01 = Actor 01
AUDIO_FILE = "dataset/Actor_01/03-01-06-01-02-01-01.wav"

# ── librosa.load() ───────────────────────────────────────────
# This is the most important line here.
# librosa.load() reads the .wav file from disk and converts it
# into a numpy array of numbers that Python can work with.
#
# It returns TWO things at once (this is called "tuple unpacking"):
#   audio       → a long array like [-0.001, 0.003, -0.002, ...]
#                 Each number = air pressure at one moment in time
#   sample_rate → how many of those numbers exist per second
#                 (e.g. 48000 means 48,000 measurements per second)
#
# sr=None means: "keep the original sample rate, don't change it"
audio, sample_rate = librosa.load(AUDIO_FILE, sr=None)

# ── Printing information about the audio ─────────────────────
# print() shows text in the terminal
# The "=" * 45 creates a line of 45 equal signs: =====================
print("=" * 45)
print("  AUDIO FILE EXPLORATION")
print("=" * 45)

# f"..." is called an f-string — it lets you put variables
# directly inside a string using curly braces {variable}
print(f"\nSample Rate  : {sample_rate} Hz")

# len() counts how many items are in a list/array
print(f"Total Samples: {len(audio)}")

# Dividing total samples by sample rate gives duration in seconds
# :.2f means "show 2 decimal places"
print(f"Duration     : {len(audio) / sample_rate:.2f} seconds")

# .shape tells you the dimensions of a numpy array
# For audio it's just (N,) — one dimension with N numbers
print(f"Array Shape  : {audio.shape}")

# .dtype tells you the type of each number
# float32 = 32-bit floating point (decimal numbers)
print(f"Data Type    : {audio.dtype}")

# .min() and .max() give the smallest and largest values
# Audio amplitude is always between -1.0 and +1.0
print(f"Min Value    : {audio.min():.4f}")
print(f"Max Value    : {audio.max():.4f}")

# Slicing: audio[:10] means "give me the first 10 items"
# This lets you see what the actual raw numbers look like
print(f"\nFirst 10 raw values (the actual sound data):")
print(audio[:10])

# ── Drawing the waveform (sound wave chart) ──────────────────
# A waveform is a graph of amplitude (loudness) over time.
# This is the classic "sound wave" shape you've seen before.
print("\nGenerating waveform chart...")

# plt.figure() creates a new blank chart
# figsize=(12, 4) sets width=12 inches, height=4 inches
plt.figure(figsize=(12, 4))

# librosa.display.waveshow() draws the waveform
# It needs the audio array AND sample_rate to calculate time on x-axis
librosa.display.waveshow(audio, sr=sample_rate)

# Add labels to make the chart readable
plt.title("Sound Wave — Fearful Speech (Actor 01)")
plt.xlabel("Time (seconds)")   # x-axis = time
plt.ylabel("Amplitude")        # y-axis = loudness

# tight_layout() prevents axis labels from being cut off at edges
plt.tight_layout()

# Save the chart as an image file in the project folder
plt.savefig("waveform.png")

# Show the chart in a popup window
plt.show()

print("Chart saved as 'waveform.png' in your project folder.")
print("=" * 45)
print("Step 1 complete! You can now see audio as raw numbers.")
