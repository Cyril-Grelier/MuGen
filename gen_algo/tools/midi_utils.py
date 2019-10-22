from midiutil import MIDIFile


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

    for i, bar in enumerate(indiv):
        for note in bar.bit.keys:
            print(note)
            MyMIDI.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)

    with open("major-scale.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)


# TODO: Try with mingus to playback midi files directly into PyCharm
# import pygame

'''
def play_music(music_file):
    """
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    """
    clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print ("Music file %s loaded!" % music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % music_file, pygame.get_error())
        return
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        # check if playback has finished
        clock.tick(30)



midi_file = '../resources/midi_example.mid'
freq = 44100    # audio CD quality
bitsize = -16   # unsigned 16 bit
channels = 2    # 1 is mono, 2 is stereo
buffer = 1024    # number of samples
pygame.mixer.init(freq, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(0.8)
try:
    play_music(midi_file)
except KeyboardInterrupt:
    # if user hits Ctrl/C then exit
    # (works only in console mode)
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit


pygame.mixer.init()
midi_file = '../resources/midi_example.mid'
pygame.mixer.music.load(midi_file)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.wait(1000)
print ("Done!")

'''
'''
wav_file = "0.wav"

import pygame
import time

pygame.init()
pygame.mixer.init()
sounda = pygame.mixer.Sound(wav_file)

sounda.play()
time.sleep(20)

'''
