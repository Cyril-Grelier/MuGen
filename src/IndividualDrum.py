import math
from random import sample, uniform, randint
import os
import numpy as np
from algo_gen.classes import Individual, Gene
from midiutil import MIDIFile
import pretty_midi
import random
from src.Convolutional_VAE.cvae_evaluator import CVae
from copy import deepcopy

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
        return str(self)


class IndividualDrum(Individual):
    _count = 0
    vae =CVae()

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
                if key_to_check.bit.timestamp <= key.bit.timestamp <= (key_to_check.bit.timestamp + key_to_check.bit.duration):
                    return False
        return True

    def mutate(self):

        for key in self.sequence:
            #print("key",key)
            # probability to switch a key

            '''
            if random.random() > 1 / len(self.sequence):
                
                change_pitch = random.randint(-1, 1)
                if self.check_collision(key, change_pitch, self.sequence) and 49 <= key.bit.pitch + change_pitch <= 58:
                    key.bit.pitch += change_pitch
                    #print("MUTATE KEY")
                self.sequence.remove(key)
            '''

            if random.random()>0.1:
                self.sequence.remove(key)
                self.generate_note()


            if random.random() > 1 / len(self.sequence):
                if random.random()>0.5:
                    if key.bit.timestamp>0.5:
                        key.bit.timestamp -= 0.01

                else:
                    if key.bit.timestamp< 7.5:
                        key.bit.timestamp += 0.01

    def crossover(self, other):
        fc = IndividualDrum(self.parameters, empty=True)
        sc = IndividualDrum(self.parameters, empty=True)
        fc.sequence = deepcopy(self.sequence)
        sc.sequence = deepcopy(other.sequence)
        return fc, sc


    def __init__(self, parameters, empty=False):
        super().__init__(parameters)
        IndividualDrum._count += 1
        self.ind = IndividualDrum._count
        if not empty:
            self.generate_seq()


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
            # print(note)
            my_midi.addNote(track, channel, note.bit.pitch, note.bit.timestamp, note.bit.duration, volume)

        with open(repertory + file, "wb") as output_file:
            my_midi.writeFile(output_file)

    def generate_note(self):

        allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
        #new_note = Note(sample(allowed_pitch, 1)[0], round_down(round(uniform(0, 7.75), 2), 0.25), 0.25) QUANTIZED
        new_note = Note(random.sample(allowed_pitch, 1)[0], round(random.uniform(0, 7.75), 2), 0.25) #UNQUANTIZED
        if new_note not in self.sequence:
            self.sequence.append(GeneDrum(new_note))




    def generate_seq(self):

        # Create a PrettyMIDI object
        #pm = pretty_midi.PrettyMIDI()


        max_number_of_notes = 100
        number_of_notes = randint(20, max_number_of_notes)
        for x in range(number_of_notes):
            self.generate_note()

        self.create_midi_file()

        '''
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
            self.sequence.append(GeneDrum(   ))
        # print(len(self.sequence))
        # b = np.array(self.sequence)
        # print(b.shape)
        '''
    def fitness(self):

        self.create_midi_file()
        repertory = "output/"
        file = repertory + str(self.ind) + ".mid"

        return -abs(self.vae.get_distance(file, self.ind ))


    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a.bit != b.bit:
                return False
        return True

    def __repr__(self):
        #r = f"I: {self.fitness()}"
        #for g in self.sequence:
        #    r += f'\n\t{g.bit}'
        r = str(self.ind)
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
    #print(f'i.fitness() : {i.fitness()}')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-n', '--name')
#     parser.add_argument('-p', '--path')
#     args = parser.parse_args()
#     # print(args.name)
#     # print(args.path)
#     generated_seq = generate_sequence()
#     convert_to_midi_seq(generated_seq, args.name, args.path)
