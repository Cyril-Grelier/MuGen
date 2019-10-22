import random

from gen_algo.individuals.automate_bar_generator import create_automate


class Note:
    def __init__(self, pitch=None, timestamp=None, duration=None):
        self.pitch = random.randrange(0, 128) if not pitch else pitch
        self.timestamp = random.randint(0, 4) if not timestamp else timestamp
        rand = random.randint(self.timestamp, 4) - self.timestamp
        self.duration = rand if rand != 0 else 1 if not duration else duration

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Pitch : {self.pitch} [t:{self.timestamp} d:{self.duration}]'


class Bar:
    def __init__(self):
        self.keys = []
        self.generate()
        print(self.keys)

    def generate(self):
        automate = create_automate()
        while (automate.has_finished() == False):
            self.add_key(automate.next_state())

    def add_key(self, key):
        self.keys.append(key)
