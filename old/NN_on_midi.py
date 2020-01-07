from datetime import datetime

import numpy as np
import pandas as pd
import pretty_midi
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential

path = '../ressources/midi_files/'
filepath = '../ressources/dataset.csv'
metadata = pd.read_csv(filepath)

features = []

# Iterate through each midi file and extract the features
for index, row in metadata.iterrows():
    path_midi_file = path + str(row["File"])
    class_label = float(row["Score"]) / 100
    midi_data = pretty_midi.PrettyMIDI(path_midi_file)
    for instrument in midi_data.instruments:
        instrument.is_drum = False
    if len(midi_data.instruments) > 0:
        data = midi_data.get_piano_roll(fs=8)
        data.resize(3968)
        result = np.where(data == 80)
        features.append([data, class_label])

# Convert into a Panda dataframe
featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features & labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# split the dataset

print(X.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_labels = 1
filter_size = 2

# Model
model = Sequential()

model.add(Dense(256, input_shape=(3968,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
# model.add(Activation('relu'))

model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 20
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='weights.best.basic_mlp.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
          callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])


def extract_feature(file_name):
    try:
        midi_data = pretty_midi.PrettyMIDI(file_name)
        for instrument in midi_data.instruments:
            instrument.is_drum = False
        if len(midi_data.instruments) > 0:
            data = midi_data.get_piano_roll(fs=8)
            data.resize(3968)
        return np.array([data])

    except Exception as e:
        print("Error parsing file : ", file_name)
        return None, None

    return np.array([data])


def print_prediction(file_name):
    prediction_feature = extract_feature(file_name)
    predicted_vector = model.predict(prediction_feature)
    print(predicted_vector)

# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_8.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_12.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_16.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_3.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/random_8.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/random_4.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/random_5.mid'
# print_prediction(file_to_test)
#
# file_to_test = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/random_3.mid'
# print_prediction(file_to_test)
