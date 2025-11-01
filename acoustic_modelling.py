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
import matplotlib.pyplot as plt
import features

name_file = open("NAMES.txt")
names = [name.strip().lower() for name in name_file.readlines()]

data = []
labels = []
npy_files = glob.glob("data/features/audio/*.npy")
MAX_FRAMES = max([np.load(npy).shape[1] for npy in npy_files])
for mfcc_file in sorted(npy_files):
    mfcc_data = np.load(mfcc_file)
    mfcc_data = np.pad(mfcc_data, ((0, 0), (0, MAX_FRAMES - mfcc_data.shape[1])))
    data.append(mfcc_data)
    stem_name = Path(os.path.basename(mfcc_file)).stem
    label = stem_name[:-3]
    print(label)
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
    num_classes = len(names)
    model = Sequential()
    model.add(InputLayer(input_shape=(42, 298, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(512)) # originally 256
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


model = create_model()

training = False
if training:

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])
    model.summary()

    EPOCHS = 25
    BATCH_SIZE = 16
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    model.save_weights('model.weights.h5')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
else:
    model.load_weights('model.weights.h5')
    '''
    mfcc = features.wav_to_mfcc('data/audio/test/test.wav')
    sd.play(sf.read('data/audio/test/test.wav.wav')[0],samplerate=16000)
    sd.wait()
    mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    p = model.predict(np.array([mfcc]))
    print("prediction: ", names[p.argmax()])
    '''


predicted_probs = model.predict(X_test, verbose=0)
predicted = np.argmax(predicted_probs, axis=1)
actual = np.argmax(y_test, axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')

predicted_prob = model.predict(np.expand_dims(X_test[0, :, :], axis=0), verbose=0)
predicted_id = np.argmax(predicted_prob, axis=1)
predicted_class = LE.inverse_transform(predicted_id)
print(predicted_class)

confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), predicted)
confusion_matrix = confusion_matrix / np.max(confusion_matrix)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix,display_labels=names)
cm_display.plot()
plt.show()