import os
import os.path
import tempfile
import numpy as np
from magenta.models.rl_tuner import rl_tuner_ops

from magenta.models.rl_tuner import note_rnn_loader
from magenta.models.rl_tuner import rl_tuner
import matplotlib
import matplotlib.pyplot as plt  # pylint: disable=unused-import
import tensorflow.compat.v1 as tf


class RLTunerTest():

    def setUp(self):
        self.output_dir = "/tmp/RL"
        self.checkpoint_dir = "/tmp/RL"
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        note_rnn = note_rnn_loader.NoteRNNLoader(
            graph, scope='test', checkpoint_dir=None)
        note_rnn.initialize_new(self.session)
        with graph.as_default():
            saver = tf.train.Saver(var_list=note_rnn.get_variable_name_dict())
            saver.save(
                self.session,
                "/tmp/RL/model.ckpt")

    def testRewardNetwork(self):
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)
        note_rnn = note_rnn_loader.NoteRNNLoader(
            graph, scope='test', checkpoint_dir=None) #, midi_primer='/tmp/RL/nice.mid')
        note_rnn.initialize_new(self.session)
        #print("NOTESEQ", note_rnn.primer)
        #print("AFER")

        rlt = rl_tuner.RLTuner(self.output_dir, note_rnn_checkpoint_dir=self.checkpoint_dir)

        initial_note = rlt.prime_internal_models()
        print("INITIAL NOTE", initial_note)
        print("FIRST SCORE", rlt.reward_key(initial_note))

        action = rlt.action(initial_note, 100, enable_random=True)
        print("ACTION  CHOSEN ", action[0])
        print("ACTION  REWARD ", action[1])
        print("ACTION  NEXT OBS ", action[2])
        print("FINAL", rlt.reward_key(action[2]))

        print("ONE HOT CREATED")
        x = np.array(rl_tuner_ops.make_onehot([10], 24)).flatten()




        print(x)
        print("FINAL", rlt.reward_key(x))

        last_observation = rlt.prime_internal_models()
        rlt.num_notes_in_melody = 12
        rlt.reset_composition()

        for _ in range(rlt.num_notes_in_melody):
            _, new_observation, reward_scores = rlt.action(
                last_observation,
                0,
                enable_random=False,
                sample_next_obs=False)

            music_theory_reward = rlt.reward_music_theory(new_observation)

            #music_theory_rewards = music_theory_reward * rlt.reward_scaler
            print(music_theory_reward)
            print(new_observation)
            rlt.composition.append(np.argmax(new_observation))
            rlt.beat += 1
            last_observation = new_observation
        print("num note", rlt.num_notes_in_melody)

        rlt.reset_composition()
        final = []
        test = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        final.append(test)

        avg = []
        print("BEGIN")
        for x in final:
            reward = rlt.reward_music_theory(x)
            print(x,reward )
            rlt.composition.append(np.argmax(x))
            rlt.beat += 1
            avg.append( reward)
            rlt.music_theory_reward_last_n += reward * rlt.reward_scaler
        print("AVG", np.mean(avg))


        rlt.reset_composition()
        final = []
        test = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        final.append(test)
        test = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)
        test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        final.append(test)
        test = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        final.append(test)


        print("SECOND")
        avg = []
        for x in final:
            reward = rlt.reward_music_theory(x)
            print(x,reward )
            rlt.composition.append(np.argmax(x))
            rlt.beat += 1
            avg.append( reward)
            rlt.music_theory_reward_last_n += reward * rlt.reward_scaler
        print("AVG", np.mean(avg))




if __name__ == '__main__':
    rl = RLTunerTest()
    rl.setUp()
    rl.testRewardNetwork()

