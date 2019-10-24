import pygame
from midiutil import MIDIFile


def convert_to_midi(indiv, file, repertory="output/"):
    track = 0
    channel = 0
    tempo = 60  # In BPM
    volume = 100  # 0-127, as per the MIDI standard

    my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created
    # automatically)
    my_midi.addTempo(track, 0, tempo)

    for i, bar in enumerate(indiv.sequence):
        for note in bar.bit:
            # print(note)
            my_midi.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)

    with open(repertory + file, "wb") as output_file:
        my_midi.writeFile(output_file)


def play_midi_file(file):
    pygame.mixer.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pass
