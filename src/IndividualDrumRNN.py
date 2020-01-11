import math
import random
from copy import deepcopy
from random import sample, uniform, randint

import numpy as np
from algo_gen.classes import Individual, Gene
from midiutil import MIDIFile

from src.data import get_drum
from src.models import get_model


def round_down(x, a):
    return math.floor(x / a) * a


class Note:
    def __init__(self, pitch=None, timestamp=None, duration=None):
        self.pitch = pitch
        self.timestamp = timestamp
        self.duration = duration

    def __eq__(self, obj):
        return isinstance(obj, Note) and obj.timestamp == self.timestamp and obj.pitch == self.pitch

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Pitch : {self.pitch} [t:{self.timestamp} d:{self.duration}]'


class GeneDrum(Gene):

    def __init__(self, notes):
        super().__init__()
        self.bit = notes

    def mutate(self):
        return "BAD MUTATE FROM GENEDRUM"

    def __str__(self):
        return str(self.bit)

    def __repr__(self):
        return ""


class IndividualDrum(Individual):
    _count = 0
    model = get_model('src/rnn_10_classes.h5')  # ("src/weights_rnn.h5")

    def __init__(self, parameters, empty=False):
        super().__init__(parameters)
        IndividualDrum._count += 1
        self.ind = IndividualDrum._count
        if not empty:
            self.generate_seq()

    def crossover(self, other):
        fc = IndividualDrum(self.parameters, empty=True)
        sc = IndividualDrum(self.parameters, empty=True)
        fc.sequence = deepcopy(self.sequence)
        sc.sequence = deepcopy(other.sequence)
        return fc, sc

    def mutate(self):
        # self.generate_note()
        for key in self.sequence:
            # self.generate_note()
            if random.random() > 1 / len(self.sequence):
                if random.random() > 0.5:
                    if key.bit.timestamp > 0.5:
                        key.bit.timestamp -= 0.1
                else:
                    if key.bit.timestamp < 7.5:
                        key.bit.timestamp += 0.1
            # if random.random() > 0.5:
            #     self.sequence.remove(key)
            # else:
            #     self.generate_note()

    def create_midi_file(self):
        track = 0
        channel = 9
        tempo = 120  # In BPM
        volume = 100  # 0-127, as per the MIDI standard
        my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
        my_midi.addTempo(track, 0, tempo)
        my_midi.addProgramChange(0, 10, 0, 0)
        my_midi.tracks[0].addChannelPressure(0, 4, 0)

        repertory = "output/"
        file = str(self.ind) + ".mid"
        for note in self.sequence:
            my_midi.addNote(track, channel, note.bit.pitch, note.bit.timestamp, note.bit.duration, volume)

        with open(repertory + file, "wb") as output_file:
            my_midi.writeFile(output_file)

    def generate_note(self):
        allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
        new_note = Note(sample(allowed_pitch, 1)[0], round_down(round(uniform(0, 7.75), 2), 0.25),
                        0.25)
        if new_note not in self.sequence:
            self.sequence.append(GeneDrum(new_note))

    def generate_seq(self):
        max_number_of_notes = 100
        number_of_notes = randint(20, max_number_of_notes)
        for x in range(number_of_notes):
            self.generate_note()

        self.create_midi_file()

    def fitness(self):
        # class
        self.create_midi_file()
        repertory = "output/"
        file = repertory + str(self.ind) + ".mid"
        data = get_drum(file)
        if type(data) == type(None):
            return 0
        prediction = self.model.predict(np.stack([data.astype(dtype=float)]))
        index_max = np.argmax(prediction)
        # pred = [0, 25, 50, 75, 100][index_max]
        pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][index_max]
        return pred

    def fitness_reg(self):
        self.create_midi_file()
        repertory = "output/"
        file = repertory + str(self.ind) + ".mid"
        data = get_drum(file)
        prediction = self.model.predict(np.stack([data.astype(dtype=float)]))
        if prediction:
            prediction = prediction[0][0]
        return prediction * 100

    def overlapped_keys(self, key_to_check, bars):
        overlapped = []
        for key in bars:
            if key_to_check.pitch != key.pitch:
                if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                    overlapped.append(key)
                    # print("key ", key_to_check, " overlapped by ", key )
        return overlapped

    def check_collision(self, key_to_check, changed_pitch, bars):
        for key in bars:
            if (key_to_check.bit.pitch + changed_pitch) == key.bit.pitch:
                if key_to_check.bit.timestamp <= key.bit.timestamp <= (
                        key_to_check.bit.timestamp + key_to_check.bit.duration):
                    return False
        return True

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a.bit != b.bit:
                return False
        return True

    def __repr__(self):
        # r = f"I: {self.fitness()}"
        # for g in self.sequence:
        #    r += f'\n\t{g.bit}'
        r = str(self.ind)
        return r

    def __hash__(self):
        r = 0
        for _ in self.sequence:
            r += randint(1, 100)
        return r
