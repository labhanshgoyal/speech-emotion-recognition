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
#def load_state(test_size=0.25):
 #   x,y = [], []

#    for f