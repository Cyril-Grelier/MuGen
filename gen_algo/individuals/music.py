import random

from midiutil import MIDIFile

from gen_algo.individuals.individual import Individual, Gene


class Note:

    def __init__(self):
        self.degree = random.randrange(0, 128)
        self.time = random.randint(0, 4)
        rand = random.randint(self.time, 4) - self.time
        self.duration = rand if rand != 0 else 1

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Pitch : {self.degree} [t:{self.time} d:{self.duration}]'


class GeneMusic(Gene):

    def __init__(self):
        super().__init__()
        self.bit = Note()

    def mutate(self):
        self.bit = Note()

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualMusic(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for _ in range(parameters['chromosome size']):
            self.sequence.append(GeneMusic())

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


def convert_to_midi(indiv):
    """

    :param indiv:
    :type indiv: IndividualMusic
    :return: 
    """

    track = 0
    channel = 0
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(track, 0, tempo)

    for i, n in enumerate(indiv):
        MyMIDI.addNote(track, channel, n.bit.degree, n.bit.time, n.bit.duration, volume)

    with open("major-scale.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)


if __name__ == '__main__':
    i = IndividualMusic({'chromosome size': 10})
    print(i)
    convert_to_midi(i)
