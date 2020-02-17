import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, LSTM
from tensorflow.keras.layers import TimeDistributed, GRU, Embedding, BatchNormalization
from tensorflow.keras.models import Sequential
import seaborn as sns
sns.set(style="ticks")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_model(model_name):
    return tf.keras.models.load_model(model_name)


def predict(model, data):
    return model.predict(data)


def cnn1(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2], 1)
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=input_shape))
    # model.add(Dropout(0.4))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(400, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(len(diff_classes), activation='sigmoid'))
    return model


def nn_feat_1(input, diff_classes):
    model = Sequential()
    model.add(Dense(100, input_dim=input[0].shape[0], kernel_initializer='normal', activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(diff_classes), activation='sigmoid'))  # , kernel_initializer='normal'))
    return model


def nn_midi():
    # Model
    model = Sequential()

    model.add(Dense(256, input_shape=(3968,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(11))
    # model.add(Activation('relu'))
    return model


def rnn1(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(128,
                   return_sequences=False,
                   input_shape=input_shape,
                   unroll=True))
    model.add(Dense(len(diff_classes)))
    model.add(Activation("softmax"))
    return model


def rnn2(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(
        Embedding(input_dim=input_shape[0], output_dim=input_shape[0], input_length=input_shape[1], mask_zero=False))
    model.add(GRU(units=256, return_sequences=True, activation="tanh", input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(diff_classes), activation='softmax')))
    return model


def rnn3(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=False, recurrent_dropout=0.3))
    # model.add(LSTM(512))
    # model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(len(diff_classes)))
    model.add(Activation('sigmoid'))
    return model


def rnn4(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(800, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(800, return_sequences=True))
    model.add(LSTM(400, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(200, activation='sigmoid'))
    # model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(len(diff_classes), activation='sigmoid'))
    return model


def rnn5(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(512, input_shape=input_shape, recurrent_dropout=0.3, return_sequences=True))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(len(diff_classes), activation='sigmoid'))
    return model


def rnn6(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(800, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(400, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(len(diff_classes)))
    model.add(Activation('sigmoid'))
    return model


def rnn7(network_input, diff_classes):
    network_input_shape = network_input.shape
    input_shape = (network_input_shape[1], network_input_shape[2])
    model = Sequential()
    model.add(LSTM(800, input_shape=input_shape, return_sequences=True))
    # model.add(LSTM(800, return_sequences=True))
    model.add(LSTM(400, return_sequences=False))
    model.add(Dense(200, activation='tanh'))
    # model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(len(diff_classes), activation='sigmoid'))
    return model


def plot_histo(what, log):  # 'loss' , 'accuracy', 'val_loss', 'val_accuracy'

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


def plot_conf_matrix(conf_arr, diff_classes):
    sum = conf_arr.sum()
    conf_arr = conf_arr * 100.0 / (1.0 * sum)

    df_cm = pd.DataFrame(conf_arr,
                         index=diff_classes,
                         columns=diff_classes)

    fig = plt.figure()

    plt.clf()

    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    cmap = sns.cubehelix_palette(light=1, as_cmap=True)

    res = sns.heatmap(df_cm, annot=True, vmin=0.0, vmax=20.0, fmt='.1f', cmap=cmap)

    res.invert_yaxis()

    # plt.yticks([0.5, 1.5, 2.5], ['Dog', 'Cat', 'Rabbit'], va='center')

    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')

    plt.close()
