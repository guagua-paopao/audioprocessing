import glob
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import features

name_file = open("NAMES.txt")
names = [name.strip().lower() for name in name_file.readlines()]

data = []
labels = []
MAX_FRAMES = 298
for mfcc_file in sorted(glob.glob("data/features/audio/*.npy")):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, MAX_FRAMES - mfcc_data.shape[1])))
    data.append(mfcc_data)
    stem_name = Path(os.path.basename(mfcc_file)).stem
    label = stem_name[:-3]
    labels.append(label)
labels = np.array(labels)
data = np.array(data)
data = data / max(abs(np.min(data)), np.max(data))  # keep the data between 0 and 1

LE = LabelEncoder()
LE.fit(names)
labels = to_categorical(LE.transform(labels))

X_train, X_tmp, y_train, y_tmp = train_test_split(data, labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)


def create_model():
    num_classes = 20
    model = Sequential()
    model.add(InputLayer(input_shape=(42, 298, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


model = create_model()
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
model.summary()

EPOCHS = 25
BATCH_SIZE = 32
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
model.save_weights('model.weights.h5')

test_rec = sd.rec(3 * 16000, samplerate=16000, channels=1)
sf.write('data/test/test.wav', test_rec, 16000)
test_mfcc = features.wav_to_mfcc("data/test/test.wav")

predicted_prob = model.predict(X_test, verbose=0)
predicted = np.argmax(predicted_prob, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(accuracy)
