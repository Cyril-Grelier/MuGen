import random
import statistics

from gen_algo.individuals.individual import Individual, Gene
from gen_algo.individuals.music_representation import Bar
from gen_algo.tools.midi_utils import convert_to_midi


def overlapped_keys(key_to_check, bars):
    overlapped = []
    for key in bars:
        if key_to_check.pitch != key.pitch:
            if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                overlapped.append(key)
                # print("key ", key, " overlap ", key_to_check )
    return overlapped

def check_collision(key_to_check,changed_pitch, bars):
    for key in bars:
        if key_to_check.pitch+changed_pitch == key.pitch:
            if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                return False
    return True


class GeneMusic(Gene):

    def __init__(self, index):
        super().__init__()
        self.bit = Bar(index * 4)

    def mutate(self):
        for key in self.bit.keys:
            #probability to switch a key
            if random.random()>1/len(self.bit.keys):
                change_pitch = random.randint(-1,1)
                if (check_collision(key,change_pitch, self.bit.keys) and key.pitch+change_pitch >=49 and key.pitch+change_pitch <= 58):
                    key.pitch += change_pitch
                #else:
                    #print("COLLISION")

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualMusic(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for index in range(parameters['chromosome size']):
            self.sequence.append(GeneMusic(index))

        print(self.fitness())

    def fitness(self):
        total_score = 0
        number_of_keys = 0
        for bars in self.sequence:
            score_bar= 0
            array_score_key = []
            # print("Mesure: ", bars)
            for key in bars.bit.keys:
                number_of_keys+=1
                # print("\t",key)
                for overlapped_key in overlapped_keys(key, bars.bit.keys):
                    #print(overlapped_key.pitch)
                    #print(key.pitch)
                    array_score_key.append( 12- abs(overlapped_key.pitch-key.pitch))

        print(array_score_key)
        total_score = statistics.mean(array_score_key)

        return total_score/number_of_keys

        # return random.random()

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
            r += random.randint(1, 100)
        return r


if __name__ == '__main__':
    i = IndividualMusic({'chromosome size': 4})
    # print(i)
    convert_to_midi(i, "coucou.mid")
