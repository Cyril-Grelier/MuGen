import random
import statistics

from gen_algo.individuals.individual import Individual, Gene
from gen_algo.individuals.music_representation import generate
from gen_algo.tools.midi_utils import convert_to_midi


def overlapped_keys(key_to_check, bars):
    overlapped = []
    for key in bars:
        if key_to_check.pitch != key.pitch:
            if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                overlapped.append(key)
                # print("key ", key, " overlap ", key_to_check )
    return overlapped


def check_collision(key_to_check, changed_pitch, bars):
    for key in bars:
        if (key_to_check.pitch + changed_pitch) == key.pitch:
            if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
                return False
    return True


class Bar(Gene):

    def __init__(self, index):
        super().__init__()
        # bit == keys
        self.bit = []
        self.generate(index*4)

    def generate(self, index):
        # automate = automate_bar_generator.create_automate()
        # while (automate.has_finished() == False):
        #     self.add_key(automate.next_state())
        # total_number_of_note = 12
        # for note in range(total_number_of_note):
        #    self.add_keys(generate(index, note + 48, total_number_of_note))
        keys = list(range(1, 13))
        new_list = random.sample(keys, 3)

        for note in new_list:
            self.add_keys(generate(index, note + 48, 0))

    def add_keys(self, keys):
        for key in keys:
            self.add_key(key)

    def add_key(self, key):
        self.bit.append(key)

    def mutate(self):
        for key in self.bit:
            # probability to switch a key
            if random.random() > 1 / len(self.bit):
                change_pitch = random.randint(-1, 1)
                if check_collision(key, change_pitch, self.bit) and 49 <= key.pitch + change_pitch <= 58:
                    key.pitch += change_pitch

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.bit)


class IndividualMusic(Individual):

    def __init__(self, parameters):
        super().__init__(parameters)
        for index in range(parameters['chromosome size']):
            self.sequence.append(Bar(index))

        print(self.fitness())

    def fitness(self):
        total_score = 0
        number_of_keys = 0
        array_score_key = []

        for gene in self.sequence:
            for key in gene.bit:
                number_of_keys += 1
                for overlapped_key in overlapped_keys(key, gene.bit):
                    array_score_key.append(12 - abs(overlapped_key.pitch - key.pitch))

        print(array_score_key)
        total_score = statistics.mean(array_score_key)

        return total_score / number_of_keys

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
    convert_to_midi(i, "coucou.mid", "../../output/")
