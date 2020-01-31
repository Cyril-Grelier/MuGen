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
import tensorflow as tf
from magenta.models.rl_tuner import rl_tuner_ops

import old.automate_bar_generator  as _automate

from src.Note import  Note



from magenta.models.rl_tuner import note_rnn_loader
from magenta.models.rl_tuner import rl_tuner

import src.reward as reward




def round_down(x, a):
    return math.floor(x / a) * a



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

    #reward = reward.RLTunerTest()
    #reward.setUp()



    graph = tf.Graph()
    session = tf.Session(graph=graph)
    note_rnn = note_rnn_loader.NoteRNNLoader(graph, scope='test', checkpoint_dir="/Users/Cyril_Musique/Documents/Cours/M2/MuGen/src/")  # , midi_primer='/tmp/RL/nice.mid')
    #note_rnn = note_rnn_loader.NoteRNNLoader(graph, scope='test',
    #                                         checkpoint_dir=None)  # , midi_primer='/tmp/RL/nice.mid')
    note_rnn.initialize_new(session)
    with graph.as_default():
        saver = tf.train.Saver(var_list=note_rnn.get_variable_name_dict())
        saver.save(session,"/tmp/RL/")

    rlt = rl_tuner.RLTuner("/Users/Cyril_Musique/Documents/Cours/M2/MuGen/output", note_rnn_checkpoint_dir="/Users/Cyril_Musique/Documents/Cours/M2/MuGen/src/")
    #rlt = rl_tuner.RLTuner("/Users/Cyril_Musique/Documents/Cours/M2/MuGen/output",
    #                       note_rnn_checkpoint_dir=None)

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

        """
        for key in self.sequence:

            RL = True

            if not RL:
                #print("key",key)
                # probability to switch a key


                #if random.random() > 1 / len(self.sequence):
                #    change_pitch = random.randint(-1, 1)
                #    if self.check_collision(key, change_pitch, self.sequence) and 49 <= key.bit.pitch + change_pitch <= 58:
                #        key.bit.pitch += change_pitch
                #        #print("MUTATE KEY")
                #    self.sequence.remove(key)
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
            if RL:

                #random_note = random.randrange(0, self.number_of_notes - 1)
                #change_pitch = random.randint(-3, 3)



                if key.bit.pitch + change_pitch < 24 and key.bit.pitch + change_pitch > 0:
                    key.bit.pitch += change_pitch

                if random.random() > 1 / len(self.sequence):
                    change_pitch = random.randint(-3,3)
                    if key.bit.pitch + change_pitch<24 and key.bit.pitch + change_pitch>0:
                        key.bit.pitch += change_pitch

                #if random.random()>0.9:
                #    key.bit.duration/=2
                #    new_note = Note(key.bit.pitch, key.bit.timestamp+key.bit.duration, key.bit.duration)
                #    self.sequence.append(GeneDrum(new_note))
                #if random.random() > 0.5:
                #    index = self.sequence.index(key)
                #    self.sequence[index-1].bit.duration*=2
                #    self.sequence.remove(key)
        """
        if len(self.sequence)>10:
            random_note = random.randrange(0, len(self.sequence))
            change_pitch = random.randint(-7, 7)
            key = self.sequence[random_note]
            if key.bit.pitch + change_pitch < 24 and key.bit.pitch + change_pitch > 0:
                key.bit.pitch += change_pitch
            #if random.random() > 0.9:
            #    index = self.sequence.index(key)
            #    if index >0:
            #        self.sequence[index-1].bit.duration*=2
            #        self.sequence.remove(key)
        else:
            print("CANNOT")




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
        max_number_of_notes = 16 #POUR 2 MESURES
        self.length=4 #*4 = 8 MESURES
        self.number_of_notes = max_number_of_notes*self.length #randint(0, max_number_of_notes)
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
            my_midi.addNote(track, channel, note.bit.pitch+36, note.bit.timestamp, note.bit.duration, volume)

        with open(repertory + file, "wb") as output_file:
            my_midi.writeFile(output_file)

    def generate_note(self):

        #allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
        new_note = Note(random.randrange(0,24), round_down(round(uniform(0, 7.75), 2), 0.25), 0.25) #QUANTIZED MELO
        #new_note = Note(sample(allowed_pitch, 1)[0], round_down(round(uniform(0, 7.75), 2), 0.25), 0.25) #QUANTIZED
        #new_note = Note(random.sample(allowed_pitch, 1)[0], round(random.uniform(0, 7.75), 2), 0.25) #UNQUANTIZED
        if new_note not in self.sequence:
            self.sequence.append(GeneDrum(new_note))








    def generate_seq(self):


        for i in range(16):
            automate = _automate.create_automate()
            while (automate.has_finished() == False):
                self.sequence.append(GeneDrum(automate.next_state(position=i * 4)))

        # Create a PrettyMIDI object
        #pm = pretty_midi.PrettyMIDI()
        """
        RL = True
        if not RL:
            max_number_of_notes = 100
            #self.number_of_notes = randint(20, max_number_of_notes)
            for x in range(self.number_of_notes):
                self.generate_note()

        if RL:
            divide = 8/self.number_of_notes*self.length
            #print(divide)
            for i in range(self.number_of_notes):

                #self.sequence.append(GeneDrum(Note(  random.randrange(0,24),i*divide, divide  )))
                self.sequence.append(GeneDrum(Note(12, i * divide, divide)))

        #print(self.sequence)
        #self.create_midi_file()
        """
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
    def fitness(self, should_print=False):


        #self.create_midi_file()
        #repertory = "output/"
        #file = repertory + str(self.ind) + ".mid"



        RL = True
        if not RL:
            return -abs(self.vae.get_distance(file, self.ind ))
        if RL:
            '''
            self.rlt.train(num_steps=100000, exploration_period=500000)

            stat_dict = self.rlt.evaluate_music_theory_metrics(num_compositions=100)
            self.rlt.plot_rewards()
            stat3 = self.rlt.generate_music_sequence(visualize_probs=True, title='post_rl')
            print(stat3)
            exit(0)
            '''

            #print("NOTE COMPO: ")
            self.rlt.num_notes_in_melody=self.number_of_notes
            self.rlt.reset_composition()
            to_mean_note_reward = []
            to_mean_rnn = []
            for note in self.sequence:

                one_hot = np.array(rl_tuner_ops.make_onehot([note.bit.pitch], 38)).flatten()
                note_reward = self.rlt.reward_music_theory(one_hot)
                #if should_print:
                #    print(one_hot,note_reward )
                self.rlt.composition.append(np.argmax(one_hot))
                self.rlt.beat += 1
                to_mean_note_reward.append(note_reward)
                a,b,c = self.rlt.action(one_hot)


                reward_scores = np.reshape(c, (38))

                #print(to_mean_rnn)

                note_rnn_reward = self.rlt.reward_from_reward_rnn_scores(one_hot,reward_scores)
                to_mean_rnn.append(note_rnn_reward)


            #print(self.rlt.composition)
            mean_note_reward = np.mean(to_mean_note_reward)
            to_mean_rnn = np.mean(to_mean_rnn)
            return to_mean_rnn + mean_note_reward
            #print(to_mean_note_reward)
            if len(to_mean_note_reward)>8:
                worst_score = np.min(to_mean_note_reward)

            #print("mean", mean_note_reward)
            if len(to_mean_note_reward) > 8:
                #if should_print:
                #    print("Worst : ",worst_score, " mean :", mean_note_reward, " ALL ", worst_score * 0.8 + 0.2 * mean_note_reward)
                return worst_score*0.8+0.2*mean_note_reward
            else:
                return -100


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
