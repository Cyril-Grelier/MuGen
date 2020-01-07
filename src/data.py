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

    return np.asarray(list(itertools.chain(*allData)))


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
