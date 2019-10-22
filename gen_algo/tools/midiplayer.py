import pygame



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



import mingus