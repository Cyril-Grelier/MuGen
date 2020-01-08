import random
import math
from midiutil import MIDIFile
import argparse



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

def round_down(x, a):
    return math.floor(x / a) * a


def generate_sequence():
    allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
    max_number_of_notes = 100
    number_of_notes = random.randint(20,max_number_of_notes)
    seq_note = []
    for x in range(number_of_notes):
        new_note = Note(random.sample(allowed_pitch,1)[0], round_down(round(random.uniform(0,7.75),2), 0.25) , 0.25)
        #if note is not already on list:
        if new_note not in seq_note:
            seq_note.append(new_note)
    return seq_note

def convert_to_midi(seq_note,file, repertory="output/"):
    track = 0
    channel = 9
    tempo = 120  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
    my_midi.addTempo(track, 0, tempo)
    my_midi.addProgramChange(0,10,0,0)
    my_midi.tracks[0].addChannelPressure(0, 4  ,0)

    for note in seq_note:
            #print(note)
            my_midi.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)

    with open(repertory + file, "wb") as output_file:
        my_midi.writeFile(output_file)



#generated_seq = generate_sequence()
#print(generated_seq)
#convert_to_midi(generated_seq, "random.mid")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    parser.add_argument('-p', '--path')
    args = parser.parse_args()
    #print(args.name)
    #print(args.path)
    generated_seq = generate_sequence()
    convert_to_midi(generated_seq, args.name, args.path)