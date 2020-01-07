import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Activation, LSTM
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_drum(file):
    midi_data = pretty_midi.PrettyMIDI(file)
    a = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            instrument.is_drum = False
            a = instrument.get_piano_roll()[36:48]
            a[a > 0] = 1
            a = np.pad(a, [(0, 0), (0, 400 - a.shape[1])], 'constant')
            a = a.astype(dtype=bool)
            # print(a[0])
            np.savetxt(file[:-4] + ".mtr", a, fmt='%.1i')
            return a.transpose()


def prepare_data():
    morceaux = []
    out_interpolation = []

    for file in tqdm(glob.glob("../resources/interpolates_files/*.mid")):
        # print("Parsing %s" % file)
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        morceaux.append(get_drum(file))

    network_input = np.stack(morceaux)

    print(network_input.shape)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output


def load_data():
    morceaux = []
    out_interpolation = []

    for file in tqdm(glob.glob("../ressources/interpolates_files/*.mtr")):
        # print("Parsing %s" % file)
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        morceaux.append(np.loadtxt(file, dtype=bool))

    network_input = np.stack(morceaux)

    print(network_input.shape)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output

# network_input, network_output = prepare_data()
network_input, network_output = load_data()
X_train = network_input.reshape(len(network_input), network_input.shape[1], network_input.shape[2], 1)

input_shape = (network_input.shape[1], network_input.shape[2], 1)

# plt.imshow(network_input[0])
# plt.show()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten


def model1(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    return model


model = model1(input_shape)
# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model = model4(input_shape)
#
# optimizer = Adam(lr=0.01)
#
# model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
tf.keras.utils.plot_model(model, to_file='model_archi.png', show_shapes=True)

number = 1
save_directory = 'test/'
log = save_directory + 'log'
model_weights = save_directory + 'model_weights_' + str(number) + '.h5'
model_architecture = save_directory + 'model_architecture_' + str(number) + '.json'

if not os.path.isdir(save_directory):
    os.mkdir(save_directory)
if not os.path.isdir(log):
    os.mkdir(log)

mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='loss', mode='min', verbose=0)
tensorboard = TensorBoard(log_dir=log + "/{}".format(str(datetime.datetime.now()) + '_model' + str(number)))
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')

callbacks_list = [mcp_save, tensorboard, early_stopping]

model_json = model.to_json()
with open(model_architecture, "w") as json_file:
    json_file.write(model_json)
print("Model saved")

# with open(save_directory + "config.json", 'w') as conf:
#     json.dump(config, conf)

log = model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list,
                validation_split=0.1, shuffle=True)


# Plot the log
def plot_histo(what):  # 'loss' , 'accuracy', 'val_loss', 'val_accuracy'
    log_history = log.history[what]
    plt.figure(facecolor='white')
    plt.plot(np.arange(len(log_history)), log_history, marker='o', color='b', label=what)
    plt.xlabel("Epoch")
    plt.ylabel(what + " history")
    plt.title("LSTM - 500 Epochs with Early Stopping")
    log_size = len(log_history)
    log_stop = log_history[log_size - 1]
    log_text = "Early Stopping (%d, %0.3f)" % (log_size, log_stop)
    plt.text(log_size - 10, log_stop + 0.2, log_text, ha='center', color='b')
    plt.grid()
    plt.show()


plot_histo('loss')
plot_histo('accuracy')
plot_histo('val_loss')
plot_histo('val_accuracy')
