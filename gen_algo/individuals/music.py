import random

from gen_algo.individuals.individual import Individual, Gene
from gen_algo.individuals.music_representation import Bar
from gen_algo.tools.midi_utils import convert_to_midi


def overlapped_keys(key_to_check, bars):
    overlapped = []
    for key in bars:
        if key_to_check.pitch != key.pitch:
            if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                overlapped.append(key)
                print("key ", key, " overlap ", key_to_check )
    return overlapped

class GeneMusic(Gene):

    def __init__(self,index):
        super().__init__()
        self.bit = Bar(index*4)

    def mutate(self):
        self.bit = Bar()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualMusic(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for index in range(parameters['chromosome size']):
            self.sequence.append( GeneMusic(index))

        print(self.fitness())

    def fitness(self):
        total_score=0
        for bars in self.sequence:
            #print("Mesure: ", bars)
            for key in bars.bit.keys:
                #print("\t",key)
                for overlapped_key in overlapped_keys(key, bars.bit.keys):
                    total_score += overlapped_key.pitch

        return total_score




        #return random.random()

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
        for g in self.sequence:
            r += ord(g.bit)
        return r


if __name__ == '__main__':
    i = IndividualMusic({'chromosome size': 4})
    #print(i)
    convert_to_midi(i)
