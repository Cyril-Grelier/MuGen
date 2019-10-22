import automate_bar_generator

class Note:
    def __init__(self, pitch, timestamp, duration):
        self.pitch = pitch
        self.timestamp = timestamp
        self.duration = duration

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
        automate = automate_bar_generator.create_automate()
        while (automate.has_finished() == False):
            self.add_key(automate.next_state())

    def add_key(self, key):
        self.keys.append(key)
