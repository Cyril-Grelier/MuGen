from abc import ABC, abstractmethod


class Individual(ABC):

    def __init__(self, parameters):
        self.sequence = []
        self.parameters = parameters

    @abstractmethod
    def crossover(self, other):
        """

        :param other: the other individual
        :type other: Individual
        :return: 2 new Individuals
        """
        pass

    def __getitem__(self, key):
        return self.sequence[key]

    def __setitem__(self, key, value):
        self.sequence[key] = value
        return value

    @abstractmethod
    def mutation(self):
        pass

    @abstractmethod
    def fitness(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    def __str__(self):
        return repr(self)

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass
