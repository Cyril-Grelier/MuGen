import random

from gen_algo.individuals.individual import Individual


class GeneOneMax:

    def __init__(self):
        self.bit = 0

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualOneMax(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for _ in range(parameters['chromosome size']):
            self.sequence.append(GeneOneMax())

    def crossover(self, other):
        first_child = IndividualOneMax(self.parameters)
        second_child = IndividualOneMax(self.parameters)
        if self.parameters['type crossover'] == 'mono-point':
            rand = random.randint(1, len(self.sequence))
            first_child[::] = self[0:rand] + other[rand:]
            second_child[::] = other[0:rand] + self[rand:]
        elif self.parameters['type crossover'] == 'uniforme':
            for i in range(self.parameters['chromosome size']):
                first_child[i], second_child[i] = (self[i], other[i]) if random.random() <= 0.5 else (
                other[i], self[i])
        return first_child, second_child

    def mutation(self):
        if self.parameters['mutation'][0] == 'n-flip':
            self.mutation_n_flip(self.parameters['mutation'][1])
        if self.parameters['mutation'][0] == 'bit-flip':
            self.mutation_bit_flip()

    def mutation_n_flip(self, n):
        for i in random.sample(range(len(self.sequence)), n):
            self.sequence[i].bit = 1 - self.sequence[i].bit

    def mutation_bit_flip(self):
        p = 1 / len(self.sequence)
        for i in range(len(self.sequence)):
            if random.random() <= p:
                self.sequence[i].bit = 1 - self.sequence[i].bit

    def fitness(self):
        return sum(v.bit for v in self.sequence)

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        for a, b in zip(self.sequence, other.sequence):
            if a.bit != b.bit:
                return False
        return True

    def __repr__(self):
        r = ""
        for g in self.sequence:
            r += str(g.bit)
        r += f" {self.fitness()}"
        return r

    def __hash__(self):
        r = ""
        for g in self.sequence:
            r += str(g.bit)
        return int(r, 2)
