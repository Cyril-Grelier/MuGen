import math
from random import sample, uniform, randint

import numpy as np
from algo_gen.classes import Individual, Gene
from midiutil import MIDIFile
from pretty_midi import PrettyMIDI

from src.VAE import get_distance


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
        for n in self.bit:
            n = 1 - n

    def __str__(self):
        pass

    def __repr__(self):
        pass


class IndividualDrum(Individual):
    _count = 0

    def __init__(self, parameters):
        super().__init__(parameters)
        IndividualDrum._count += 1
        self.ind = IndividualDrum._count

        self.generate_seq()

    def generate_seq(self):
        repertory = "output/"
        file = str(self.ind) + ".mid"
        allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
        max_number_of_notes = 100
        number_of_notes = randint(20, max_number_of_notes)
        seq_note = []
        for x in range(number_of_notes):
            new_note = Note(sample(allowed_pitch, 1)[0], round_down(round(uniform(0, 7.75), 2), 0.25),
                            0.25)
            if new_note not in seq_note:
                seq_note.append(new_note)
        track = 0
        channel = 9
        tempo = 120  # In BPM
        volume = 100  # 0-127, as per the MIDI standard

        my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
        my_midi.addTempo(track, 0, tempo)
        my_midi.addProgramChange(0, 10, 0, 0)
        my_midi.tracks[0].addChannelPressure(0, 4, 0)

        for note in seq_note:
            # print(note)
            my_midi.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)

        with open(repertory + file, "wb") as output_file:
            my_midi.writeFile(output_file)

        midi_data = PrettyMIDI(repertory + file)
        a = None
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                instrument.is_drum = False
                a = instrument.get_piano_roll()[36:48]
                a[a > 0] = 1
                a = np.pad(a, [(0, 0), (0, 400 - a.shape[1])], 'constant')
                a = a.astype(dtype=bool)
                # a = a.transpose()
                break
        for i in range(a.shape[0]):
            self.sequence.append(GeneDrum(list(a[i])))
        # print(len(self.sequence))
        # b = np.array(self.sequence)
        # print(b.shape)

    def fitness(self):
        return get_distance("output/" + str(self.ind) + ".mid")

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a.bit != b.bit:
                return False
        return True

    def __repr__(self):
        r = f"I: {self.fitness()}"
        for g in self.sequence:
            r += f'\n\t{g.bit}'
        return r

    def __hash__(self):
        r = 0
        for _ in self.sequence:
            r += randint(1, 100)
        return r


if __name__ == '__main__':
    i = IndividualDrum({'chromosome size': 12})
    # print(i)
    # convert_to_midi(i, "coucou.mid", "../../output/")
    print(f'i.fitness() : {i.fitness()}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--name')
#     parser.add_argument('-p', '--path')
#     args = parser.parse_args()
#     # print(args.name)
#     # print(args.path)
#     generated_seq = generate_sequence()
#     convert_to_midi_seq(generated_seq, args.name, args.path)
