import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping

import src.data as d
import src.models as models
from src.models import plot_histo


# def print_prediction(file_name):
#     prediction_feature = extract_feature(file_name)
#     predicted_vector = model.predict(prediction_feature)
#     print(predicted_vector)

def train_model(model_type, data_fct, loss, optimizer, epochs, batch_size, name):
    network_input, network_output = data_fct()
    model = model_type(network_input.shape)
    compile(model, loss, optimizer)
    log = fit(model, name, network_input, network_output, epochs, batch_size)
    return log


def compile(model, loss, optimizer):
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_archi.png', show_shapes=True)


def fit(model, name, network_input, network_output, epochs, batch_size):
    save_directory = 'test/'
    log = save_directory + 'log'
    model_weights = save_directory + 'model_weights_' + str(name) + '.h5'
    model_architecture = save_directory + 'model_architecture_' + str(name) + '.json'

    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    if not os.path.isdir(log):
        os.mkdir(log)

    mcp_save = ModelCheckpoint(model_weights, save_best_only=True, monitor='loss', mode='min', verbose=0)
    tensorboard = TensorBoard(log_dir=log + "/{}".format(str(datetime.now()) + '_model' + str(name)))
    early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')

    callbacks_list = [mcp_save, tensorboard, early_stopping]

    model_json = model.to_json()
    with open(model_architecture, "w") as json_file:
        json_file.write(model_json)
    print("Model saved")

    log = model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
                    validation_split=0.1, shuffle=True)
    return log


# log = train_model(models.cnn1, d.prepare_data_cnn, loss='binary_crossentropy', optimizer=Adam(lr=0.01), epochs=200, batch_size=16, name='rnn4')
# binary_crossentropy
log = train_model(models.rnn7, d.prepare_data_midi_class, loss='binary_crossentropy', optimizer=Adam(lr=0.01),
                  epochs=200, batch_size=16, name='rnn4')

plot_histo('loss', log)
plot_histo('accuracy', log)
plot_histo('val_loss', log)
plot_histo('val_accuracy', log)

#################################################################################### CNN midi
# network_input, network_output = prepare_data_midi()
# network_input, network_output = load_data_midi()
#
# input_shape = (network_input.shape[1], network_input.shape[2], 1)
#
# # plt.imshow(network_input[0])
# # plt.show()
#
# X_train = network_input.reshape(len(network_input), network_input.shape[1], network_input.shape[2], 1)
#
# model = cnn1(input_shape)
# # compile model using accuracy to measure model performance
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # model = model4(input_shape)
# #
# # optimizer = Adam(lr=0.01)
# #
# # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.summary()
# tf.keras.utils.plot_model(model, to_file='model_archi.png', show_shapes=True)
#
#
# log = fit(model, 'cnn1', network_input, network_output, epochs=200, batch_size=32)

#################################################################################### NN midi

#
# features = get_midifeat()
# # Convert into a Panda dataframe
# featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])
#
# print('Finished feature extraction from ', len(featuresdf), ' files')
#
# # Convert features & labels into numpy arrays
# X = np.array(featuresdf.feature.tolist())
# y = np.array(featuresdf.class_label.tolist())
#
# # split the dataset
#
# print(X.shape, y.shape)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# num_labels = 1
# filter_size = 2
# model = nn_midi()
#
# model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer='adam')
# model.summary()
#
# num_epochs = 20
# num_batch_size = 32
#
# checkpointer = ModelCheckpoint(filepath='weights.best.basic_mlp.hdf5',
#                                verbose=1, save_best_only=True)
# start = datetime.now()
#
# model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test),
#           callbacks=[checkpointer], verbose=1)
#
# # Evaluating the model on the training and testing set
# score = model.evaluate(x_train, y_train, verbose=0)
# print("Training Accuracy: ", score[1])
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Testing Accuracy: ", score[1])

#################################################################################### RNN midi


# network_input, network_output = prepare_data_midi()
# # network_input, network_output = load_data_midi()
#
# # np.savetxt("network_output", network_output)
# # np.savetxt("network_input", network_input)
#
# # model = rnn1(network_input.shape)
# # model = rnn2(network_input.shape)
# model = rnn3(network_input.shape)
# # model = rnn4(network_input.shape)
# # model = rnn5(network_input.shape)
#
# optimizer = Adam(lr=0.005)
#
# loss='binary_crossentropy'
# compile(model, loss, optimizer)
#
# log = fit(model, 'rnn4', network_input, network_output, epochs=200, batch_size=32)


#################################################################################### NN features


# network_input, network_output = prepare_data_features()
# network_input, network_output = load_data_feat()
#
# network_input = network_input[:, ~np.all(network_input[1:] == network_input[:-1], axis=0)]
#
# print(len(network_input[0]))
#
# model = nn_feat_1(len(network_input[0]))
#
# optimizer = Adam(lr=0.01)
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
# model.summary()
# tf.keras.utils.plot_model(model, to_file='model_archi.png', show_shapes=True)
#
#
# log = fit(model, 'nn_feat', network_input, network_output, epochs=200, batch_size=32)
