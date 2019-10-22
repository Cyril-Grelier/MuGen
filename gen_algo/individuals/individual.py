from abc import ABC, abstractmethod


class Gene(ABC):

    def __init__(self):
        self.bit = None

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class Individual(ABC):

    def __init__(self, parameters):
        self.sequence = []
        self.parameters = parameters

    def __getitem__(self, key):
        return self.sequence[key]

    def __setitem__(self, key, value):
        self.sequence[key] = value
        return value

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
