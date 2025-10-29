import glob
import os
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import numpy as np

name_file = open("NAMES.txt")
names = [name.strip().lower() for name in name_file.readlines()]

data = []
labels = []
MAX_FRAMES = 94
for mfcc_file in sorted(glob.glob("features/mfccs/*.npy")):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, MAX_FRAMES - mfcc_data.shape[1])))
    data.append(mfcc_data)
    stem_name = Path(os.path.basename(mfcc_file)).stem
    label = stem_name[:-3]
    labels.append(label)
labels = np.array(labels)
data =np.array(data)
data = data