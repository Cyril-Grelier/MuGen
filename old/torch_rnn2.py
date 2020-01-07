import os

import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.utils.data as data
from IPython.display import FileLink
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils import data
from glob import glob
import pretty_midi
from midi.midi_utils import midiread, midiwrite

from torch import nn
from torch.nn.utils.rnn import pad_sequence

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def get_drums(midi_file):
    # midi_data = midiread(midi_file, dt=0.3)
    # drums = midi_data.piano_roll.transpose()
    # drums[drums > 0] = 1
    # return drums
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    a = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            instrument.is_drum = False
            a = instrument.get_piano_roll()[36:60]
            a[a > 0] = 1
            break
    return a


def pad_drums(drums, max_length=120000, pad_value=0):
    original_length = drums.shape[1]
    padded_drums = np.zeros((60-36, max_length))
    padded_drums[:] = pad_value
    padded_drums[:, -original_length:] = drums
    return padded_drums


class DrumsDataset(data.Dataset):

    def __init__(self, path="output/train", longest_sequence_length=None):
        self.path = path
        self.midi_files = glob(path + "/*/*.mid")
        self.longest_sequence_length = longest_sequence_length
        # TODO faire la premiere fois puis rentrer la taille direct...
        if longest_sequence_length is None:
            sequences_lengths = map(lambda filename: get_drums(filename).shape[1], self.midi_files)
            max_length = max(sequences_lengths)
            self.longest_sequence_length = max_length

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, index):
        midi_file = self.midi_files[index]
        drums = get_drums(midi_file)
        sequence_length = drums.shape[1]
        input_padded = pad_drums(drums, max_length=self.longest_sequence_length).transpose()
        out = float(midi_file.split("/")[-2])
        input = torch.FloatTensor(input_padded)
        output = torch.LongTensor([out])
        size = torch.LongTensor([sequence_length])

        # return drums_padded, output, size
        return input, output, size


def post_process_sequence_batch(batch):
    inputs, outputs, sizes = batch

    splitted_inputs = inputs.split(split_size=1)
    splitted_outputs = outputs.split(split_size=1)
    splitted_sizes = sizes.split(split_size=1)

    training_data = zip(splitted_inputs, splitted_outputs, splitted_sizes)

    training_data_sorted = sorted(training_data, key=lambda p: int(p[2]), reverse=True)

    splitted_inputs, splitted_outputs, splitted_sizes = zip(*training_data_sorted)

    inputs = torch.cat(splitted_inputs)
    outputs = torch.cat(splitted_outputs)
    sizes = torch.cat(splitted_sizes)

    inputs = inputs[:, -sizes[0, 0]:, :]
    inputs = inputs.transpose(0, 1)

    outputs = outputs[:, -sizes[0, 0]:]
    sizes = list(map(lambda x: int(x), list(sizes)))

    return inputs, outputs, sizes


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers)
        self.logits_fc = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, sizes, hidden=None):
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, sizes)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        prediction = self.logits_fc(outputs)
        return prediction, hidden


def validate(model):
    model.eval()
    full_val_loss = 0.0
    overall_sequence_length = 0.0

    for batch in valset_loader:
        processed_batch = post_process_sequence_batch(batch)
        inputs, outputs, sizes = processed_batch
        # inputs = Variable(inputs.to(device))
        prediction, _ = model(inputs, sizes)
        loss = criterion_val(prediction, outputs)
        full_val_loss += loss.item()
        overall_sequence_length += sum(sizes)
    return full_val_loss / (overall_sequence_length * 88)


trainset_loader = data.DataLoader(DrumsDataset("output/train"), batch_size=8, shuffle=True, drop_last=True)

valset_loader = data.DataLoader(DrumsDataset("output/val"), batch_size=8, shuffle=True, drop_last=True)

model = RNN(input_size=60-36, hidden_size=128, output_size=1)#.to(device)

print(model)

criterion = nn.CrossEntropyLoss()#.to(device)
criterion_val = nn.CrossEntropyLoss()#.to(device)

validate(model)
