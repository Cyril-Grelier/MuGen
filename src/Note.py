class Note:
    def __init__(self, pitch=None, timestamp=None, duration=None):
        self.pitch = pitch
        self.timestamp = timestamp
        self.duration = duration

    def __eq__(self, obj):
        return isinstance(obj, Note) and obj.timestamp == self.timestamp and obj.pitch == self.pitch

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'Pitch : {self.pitch} [t:{self.timestamp} d:{self.duration}]'

