import random


def generate(index, pitch, total_number_of_note):
    """

    :param index:
    :param pitch:
    :param total_number_of_note:
    :return:
    """

    # Pour 1 mesure : avec une resolution de 64
    # Ronde = 1
    # Blanche = 1/2
    # Noire = 1/4
    # Croche = 1/8
    # Double croche = 1/16
    # Triple croche = 1/32
    # Quadruple croche = 1/64

    resolution_per_bar = 4
    total_number_of_steps = resolution_per_bar
    time_credit = total_number_of_steps
    used_credit = 0
    seq_note = []
    # print("duree totale : ", time_credit)
    while time_credit - used_credit > 0:
        # print(index)
        new_note_start = round(used_credit + index, 1)
        new_note_duration = round(random.uniform(0.1, min(4, time_credit - used_credit)), 1)

        if random.random() > (1 - (1 / total_number_of_note) * 3):
            new_note = Note(pitch, new_note_start, new_note_duration)
            seq_note.append(new_note)

        used_credit = round(used_credit + new_note_duration, 1)
    # print(seq_note)
    # print("durée totale:", used_credit)
    return seq_note


class Note:
    def __init__(self, pitch=None, timestamp=None, duration=None):
        self.pitch = pitch
        self.timestamp = timestamp
        self.duration = duration

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Pitch : {self.pitch} [t:{self.timestamp} d:{self.duration}]'


class Bar:
    def __init__(self, index):
        self.keys = []
        self.generate(index)
        # print(self.keys)

    def generate(self, index):
        # automate = automate_bar_generator.create_automate()
        # while (automate.has_finished() == False):
        #     self.add_key(automate.next_state())
        total_number_of_note = 12
        for note in range(total_number_of_note):
            self.add_keys(generate(index, note + 48, total_number_of_note))

    def add_keys(self, keys):
        for key in keys:
            self.add_key(key)

    def add_key(self, key):
        self.keys.append(key)
