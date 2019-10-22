import random
from gen_algo.individuals.music_representation import Note, Bar
from gen_algo.individuals.individual import Individual, Gene
from  gen_algo.tools.midi_utils import convert_to_midi

class GeneMusic(Gene):

    def __init__(self):
        super().__init__()
        self.bit = Bar()

    def mutate(self):
        self.bit = Bar()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualMusic(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for _ in range(parameters['chromosome size']):
            self.sequence.append( GeneMusic())


    def fitness(self):
        return random.random()

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
    i = IndividualMusic({'chromosome size': 1})
    print(i)
    convert_to_midi(i)
