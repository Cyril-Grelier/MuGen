import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pretty_midi
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Lambda, Input
# from tensorflow.keras.losses import mse
from tensorflow.keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_coord(models, data, batch_size=128):
    encoder, decoder = models
    x_test, y_test = data

    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    # print(f'z_mean : {z_mean}')

    fig, ax = plt.subplots()
    scatter = ax.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    plt.show()
    return z_mean


def get_features_file(file, class_label):
    midi_data = pretty_midi.PrettyMIDI(file)
    a = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            instrument.is_drum = False
            a = instrument.get_piano_roll()[36:48]
            a[a > 0] = 1
            a = np.pad(a, [(0, 0), (0, 400 - a.shape[1])], 'constant')
            a = a.astype(dtype=bool)
            a.resize(4800)
            # print(a[0])
            # np.savetxt(file[:-4] + ".mtr", a, fmt='%.1i')
            return [a, class_label]
    # midi_data = pretty_midi.PrettyMIDI(file)
    #
    # for instrument in midi_data.instruments:
    #     instrument.is_drum = False
    # if len(midi_data.instruments) > 0:
    #     data = midi_data.get_piano_roll(fs=8)
    #     data.resize(3968)
    #     return [data, class_label]


def get_features_all_data():
    features = []
    path = 'ressources/dataset_csv/midi_files/'
    filepath = 'ressources/dataset_csv/dataset.csv'

    metadata = pd.read_csv(filepath)

    # Iterate through each midi file and extract the features
    for index, row in metadata.iterrows():
        path_midi_file = path + str(row["File"])
        if row["Score"] in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:  # [0,50,100]:
            # if row["Score"] != 89:
            class_label = float(row["Score"]) / 100
            features.append(get_features_file(path_midi_file, class_label))
    return features


def model(input_shape):
    print(input_shape)
    intermediate_dim = 512
    latent_dim = 2
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    # encoder.summary()
    # plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(input_shape[0], activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    # decoder.summary()
    # plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs)  # , name='vae_mlp')

    vae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # vae.summary()
    # plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

    return vae, encoder, decoder


def load_data(training, path_to_plot=""):
    if training:
        features = get_features_all_data()
    else:
        features = [get_features_file(path_to_plot, 0)]

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    # print('Finished feature extraction from ', len(featuresdf), ' files')

    # Convert features & labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())
    y = np.array(featuresdf.class_label.tolist())

    # print(X.shape, y.shape)

    if training:
        # split the dataset
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        x_train = X
        y_train = y

    # network parameters
    input_shape = (x_train.shape[1],)

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    if training:
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae, encoder, decoder = model(input_shape)

    if training:
        return vae, encoder, decoder, x_train, y_train, x_test, y_test
    else:
        return vae, encoder, decoder, x_train, y_train


def train():
    batch_size = 128
    epochs = 100
    batch_size = 128
    vae, encoder, decoder, x_train, y_train, x_test, y_test = load_data(True)
    data = (x_train, y_train)
    # train the autoencoder
    vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, x_test))
    # validation_data=(x_test, None))
    vae.save_weights('vae_midi.h5')
    models = (encoder, decoder)
    coord = get_coord(models, data, batch_size=batch_size)

    x = coord[:, 0]
    y = coord[:, 1]
    # print(x, y)

    # distance = math.sqrt(((0 - x) ** 2) + ((0 - y) ** 2))


def give_distance(model, file):
    model = tf.keras.models.load_model('vae_midi.h5')


def get_distance(file):
    batch_size = 128
    vae, encoder, decoder, x_train, y_train = load_data(False, file)
    data = (x_train, y_train)
    vae.load_weights('vae_midi.h5')

    models = (encoder, decoder)
    coord = get_coord(models, data, batch_size=batch_size)

    x = coord[:, 0]
    y = coord[:, 1]
    # print(x, y)

    distance = math.sqrt(((0 - x) ** 2) + ((0 - y) ** 2))
    # print(f'distance : {distance}')
    return distance

#
# def main():
#     print(f'distance : {get_distance("2519_60.mid")}')
#
#
# if __name__ == '__main__':
#     main()
