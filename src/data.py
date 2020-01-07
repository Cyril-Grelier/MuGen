import glob
import itertools
from glob import glob

import music21
import numpy as np
import pretty_midi
from tqdm import tqdm
import pandas as pd

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



def get_midifeat():
    path = '../resources/midi_files/'
    filepath = '../resources/dataset.csv'
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
            # plot des 0 et 1 du fichier midi
            # plt.imshow(network_input[0])
            # plt.show()
    return features


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
            np.savetxt(file[:-4] + ".mtr", a, fmt='%.1i')
            return a.transpose()





def prepare_data_midi():
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


def prepare_data_features():
    features = []
    out_interpolation = []

    for file in tqdm(glob("../resources/interpolates_files/*.mid")):
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        features.append(get_features(file))
        np.savetxt(file[:-4] + ".feat", features[-1])

    network_input = np.asarray(features)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output


def load_data_midi():
    morceaux = []
    out_interpolation = []

    for file in tqdm(glob.glob("../resources/interpolates_files/*.mtr")):
        # print("Parsing %s" % file)
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        morceaux.append(np.loadtxt(file, dtype=bool))

    network_input = np.stack(morceaux)

    print(network_input.shape)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output


def load_data_feat():
    features = []
    out_interpolation = []

    for file in tqdm(glob("../resources/interpolates_files/*.feat")):
        out_interpolation.append(float(file.split('/')[-1].split("_")[1].split(".")[0]) / 100)
        features.append(np.loadtxt(file))

    network_input = np.asarray(features)
    network_output = np.asarray(out_interpolation)
    print(network_output.shape)
    return network_input, network_output
