import numpy as np
import pretty_midi
import torch
from torch import nn
import random
from torch.utils import data

from midi.midi_utils import midiread


# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#
#         # self.rnn = nn.RNN(
#         #     input_size=12,
#         #     hidden_size=32,  # rnn hidden unit
#         #     num_layers=1,  # number of rnn layer
#         #     batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
#         # )
#         self.lstm = nn.LSTM(12, 32, 1)
#         self.out = nn.Linear(32, 1)
#
#     def forward(self, x, h):
#         h_t = torch.zeros(input.size(0), 32, dtype=torch.float32)
#         c_t = torch.zeros(input.size(0), 32, dtype=torch.float32)
#         out, h = self.lstm(x, (h_t,c_t))
#         outs = self.out(out)
#         return outs, h

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, future=0, y=None):
        outputs = []        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t = torch.zeros(input.size(-3), self.hidden_size, dtype=torch.float32)
        c_t = torch.zeros(input.size(2), self.hidden_size, dtype=torch.float32)
        for i, input_t in enumerate(input.chunk(input.size(-1), dim=0)):
            print(input_t.shape)
            h_t, c_t = self.lstm(input_t, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
            h_t, c_t = self.lstm(output, (h_t, c_t))
            output = self.linear(h_t)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def get_drum(file="generated_1.mid"):
    # midi_data = midiread(file, r=(36, 48), dt=1 / 100)  # 48 ou 60
    #
    # piano_roll = midi_data.piano_roll.transpose()
    # piano_roll[piano_roll > 0] = 1
    # return piano_roll

    midi_data = pretty_midi.PrettyMIDI(file)
    a = None
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            instrument.is_drum = False
            a = instrument.get_piano_roll()[36:48]
            a[a > 0] = 1
            break
    return a.transpose()


class Dataset(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, list_IDs, labels):
        """
        Initialization
        """
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """
        Denotes the total number of samples
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.FloatTensor(get_drum(ID).astype(np.float64))

        y = self.labels[ID]

        return X, y


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 100

from glob import glob

train_files = glob("output/train/*/*.mid")
val_files = glob("output/val/*/*.mid")
test_files = glob("output/test/*/*.mid")
print(train_files)

labels = {f: float(f.split("/")[-2]) for f in train_files + val_files + test_files}

# Datasets
partition = {"train": train_files, "validation": val_files, "test": test_files}

# Generatorspython
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels)
validation_generator = data.DataLoader(validation_set, **params)

test_set = Dataset(partition['test'], labels)
test_generator = data.DataLoader(test_set, **params)

rnn = RNN(387,12,1)
# rnn.to(device)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)  # optimize all cnn parameters
loss_func = nn.MSELoss()

h = None

# x = np.array([get_drum(train_files[0]), get_drum(train_files[1])])
# y = [labels[train_files[i]] for i in range(4)]
# print(type(x))
# # x = torch.from_numpy(x)
# print(x.shape)
# x_ = torch.from_numpy(x)
# # y = torch.FloatTensor([y])
#
# prediction, h = rnn(x_.float(), h)
#
# print(prediction.shape)

# Loop over epochs
for epoch in range(max_epochs):
    print(f'epoch {epoch}')
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        print(local_batch.shape)
        # prediction, h = rnn(local_batch, h)  # rnn output
        # !! next step is important !!
        # print(prediction.shape)
        # print(h)
        # h = h.data  # repack the hidden state, break the connection from last iteration
        #
        # loss = loss_func(prediction, local_labels.float())  # calculate loss
        # print(loss)
        # optimizer.zero_grad()  # clear gradients for this training step
        # loss.backward()  # backpropagation, compute gradients
        # optimizer.step()  # apply gradients

        import torch.optim as optim

        # create your optimizer
        optimizer = optim.SGD(rnn.parameters(), lr=0.01)

        # in your training loop:
        optimizer.zero_grad()  # zero the gradient buffers
        prediction = rnn(local_batch)
        print(h)
        h.detach_()
        loss = loss_func(prediction, local_labels.float())
        loss.backward()
        optimizer.step()  # Does the update
        hidden = h.detach()

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            print(local_batch)
            print(local_labels)

