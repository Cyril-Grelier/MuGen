import datetime
import glob
import itertools
import os
from glob import glob

import matplotlib.pyplot as plt
import music21
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard, EarlyStopping
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def get_features(file):
    features = [
        music21.features.jSymbolic.MelodicIntervalHistogramFeature,
        music21.features.jSymbolic.AverageMelodicIntervalFeature,
        music21.features.jSymbolic.MostCommonMelodicIntervalFeature,
        music21.features.jSymbolic.DistanceBetweenMostCommonMelodicIntervalsFeature,
        music21.features.jSymbolic.MostCommonMelodicIntervalPrevalenceFeature,
        music21.features.jSymbolic.RelativeStrengthOfMostCommonIntervalsFeature,
        music21.features.jSymbolic.NumberOfCommonMelodicIntervalsFeature,
        music21.features.jSymbolic.AmountOfArpeggiationFeature,
        music21.features.jSymbolic.RepeatedNotesFeature,
        music21.features.jSymbolic.ChromaticMotionFeature,
        music21.features.jSymbolic.StepwiseMotionFeature,
        music21.features.jSymbolic.MelodicThirdsFeature,
        music21.features.jSymbolic.MelodicFifthsFeature,
        music21.features.jSymbolic.MelodicTritonesFeature,
        music21.features.jSymbolic.MelodicOctavesFeature,
        music21.features.jSymbolic.DirectionOfMotionFeature,
        music21.features.jSymbolic.DurationOfMelodicArcsFeature,
        music21.features.jSymbolic.SizeOfMelodicArcsFeature,
        music21.features.jSymbolic.PitchedInstrumentsPresentFeature,
        music21.features.jSymbolic.NotePrevalenceOfPitchedInstrumentsFeature,
        music21.features.jSymbolic.VariabilityOfNotePrevalenceOfPitchedInstrumentsFeature,
        music21.features.jSymbolic.NumberOfPitchedInstrumentsFeature,
        music21.features.jSymbolic.StringKeyboardFractionFeature,
        music21.features.jSymbolic.AcousticGuitarFractionFeature,
        music21.features.jSymbolic.ElectricGuitarFractionFeature,
        music21.features.jSymbolic.ViolinFractionFeature,
        music21.features.jSymbolic.SaxophoneFractionFeature,
        music21.features.jSymbolic.BrassFractionFeature,
        music21.features.jSymbolic.WoodwindsFractionFeature,
        music21.features.jSymbolic.OrchestralStringsFractionFeature,
        music21.features.jSymbolic.StringEnsembleFractionFeature,
        music21.features.jSymbolic.ElectricInstrumentFractionFeature,
        music21.features.jSymbolic.NoteDensityFeature,
        music21.features.jSymbolic.AverageNoteDurationFeature,
        music21.features.jSymbolic.VariabilityOfNoteDurationFeature,
        music21.features.jSymbolic.MaximumNoteDurationFeature,
        music21.features.jSymbolic.MinimumNoteDurationFeature,
        music21.features.jSymbolic.StaccatoIncidenceFeature,
        music21.features.jSymbolic.AverageTimeBetweenAttacksFeature,
        music21.features.jSymbolic.VariabilityOfTimeBetweenAttacksFeature,
        music21.features.jSymbolic.AverageTimeBetweenAttacksForEachVoiceFeature,
        music21.features.jSymbolic.AverageVariabilityOfTimeBetweenAttacksForEachVoiceFeature,
        music21.features.jSymbolic.InitialTempoFeature,
        music21.features.jSymbolic.InitialTimeSignatureFeature,
        music21.features.jSymbolic.CompoundOrSimpleMeterFeature,
        music21.features.jSymbolic.TripleMeterFeature,
        music21.features.jSymbolic.QuintupleMeterFeature,
        music21.features.jSymbolic.ChangesOfMeterFeature,
        music21.features.jSymbolic.DurationFeature,
        music21.features.jSymbolic.MaximumNumberOfIndependentVoicesFeature,
        music21.features.jSymbolic.AverageNumberOfIndependentVoicesFeature,
        music21.features.jSymbolic.VariabilityOfNumberOfIndependentVoicesFeature,
        music21.features.jSymbolic.MostCommonPitchPrevalenceFeature,
        music21.features.jSymbolic.MostCommonPitchClassPrevalenceFeature,
        music21.features.jSymbolic.RelativeStrengthOfTopPitchesFeature,
        music21.features.jSymbolic.RelativeStrengthOfTopPitchClassesFeature,
        music21.features.jSymbolic.IntervalBetweenStrongestPitchesFeature,
        music21.features.jSymbolic.IntervalBetweenStrongestPitchClassesFeature,
        music21.features.jSymbolic.NumberOfCommonPitchesFeature,
        music21.features.jSymbolic.PitchVarietyFeature,
        music21.features.jSymbolic.PitchClassVarietyFeature,
        music21.features.jSymbolic.RangeFeature,
        music21.features.jSymbolic.MostCommonPitchFeature,
        music21.features.jSymbolic.PrimaryRegisterFeature,
        music21.features.jSymbolic.ImportanceOfBassRegisterFeature,
        music21.features.jSymbolic.ImportanceOfMiddleRegisterFeature,
        music21.features.jSymbolic.ImportanceOfHighRegisterFeature,
        music21.features.jSymbolic.MostCommonPitchClassFeature,
        music21.features.jSymbolic.BasicPitchHistogramFeature,
        music21.features.jSymbolic.PitchClassDistributionFeature,
        music21.features.jSymbolic.FifthsPitchHistogramFeature,
        music21.features.jSymbolic.QualityFeature,
        music21.features.native.QualityFeature,
        music21.features.native.TonalCertainty,
        music21.features.native.UniqueNoteQuarterLengths,
        music21.features.native.MostCommonNoteQuarterLength,
        music21.features.native.MostCommonNoteQuarterLengthPrevalence,
        music21.features.native.RangeOfNoteQuarterLengths,
        music21.features.native.UniquePitchClassSetSimultaneities,
        music21.features.native.UniqueSetClassSimultaneities,
        music21.features.native.MostCommonPitchClassSetSimultaneityPrevalence,
        music21.features.native.MostCommonSetClassSimultaneityPrevalence,
        music21.features.native.MajorTriadSimultaneityPrevalence,
        music21.features.native.MinorTriadSimultaneityPrevalence,
        # music21.features.native.DominantSeventhSimultaneityPrevalence,
        # music21.features.native.DiminishedTriadSimultaneityPrevalence,
        # music21.features.native.TriadSimultaneityPrevalence,
        # music21.features.native.DiminishedSeventhSimultaneityPrevalence,
        # music21.features.native.IncorrectlySpelledTriadPrevalence,
        # music21.features.native.ChordBassMotionFeature,
        # music21.features.native.ComposerPopularity,
        # music21.features.native.LandiniCadence,
        # music21.features.native.LanguageFeature,
    ]

    mf = music21.midi.MidiFile()
    mf.open(str(file))
    mf.read()
    mf.close()
    # len(mf.tracks)

    s = music21.midi.translate.midiFileToStream(mf)

    ds = music21.features.base.DataSet(classLabel="")
    ds.addFeatureExtractors(features)
    ds.addData(s)
    ds.process()
    allData = ds.getFeaturesAsList(includeClassLabel=False,
                                   includeId=False,
                                   concatenateLists=False)

    return list(itertools.chain(*allData))


def prepare_data():
    features = []
    out_interpolation = []

    for file in tqdm(glob("../resources/interpolates_files/*.mid")):
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        features.append(np.asarray(get_features(file)))
        np.savetxt(file[:-4] + ".feat", features[-1])

    network_input = np.asarray(features)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output


def load_data():
    features = []
    out_interpolation = []

    for file in tqdm(glob("../resources/interpolates_files/*.feat")):
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        features.append(np.loadtxt(file))

    network_input = np.asarray(features)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output


network_input, network_output = load_data()

network_input = network_input[:, ~np.all(network_input[1:] == network_input[:-1], axis=0)]

print(len(network_input[0]))


def model1(input_shape):
    model = Sequential()
    model.add(Dense(input_shape, input_dim=input_shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    return model


model = model1(len(network_input[0]))

optimizer = Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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
