import librosa
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
name_file = open("NAMES.txt")
names = [name.strip().lower() for name in name_file.readlines()]
mfcc_dict = {}
for filename in os.listdir("data/audio/"):
    if filename[:-7] in names and filename[-4:] == ".wav":
        speech, fs = sf.read(os.path.join("data/audio", filename))
        mfcc = librosa.feature.mfcc(y=speech, sr=fs)
        outname = os.path.join('features/mfccs/', filename[:-4] + '.npy')
        with open(outname, "wb") as f:
            np.save(f, mfcc)