# import argparse
# import contextlib
# import math
# import random
#
# from algo_gen.classes import Individual, Gene
# from midiutil import MIDIFile
#
# # from VAE.VAE import get_distance
# #
# # with contextlib.redirect_stdout(None):
# #     import pygame
#
#
# def round_down(x, a):
#     return math.floor(x / a) * a
#
#
# def generate_sequence():
#     allowed_pitch = [36, 38, 42, 46, 41, 45, 48, 51, 49]
#     max_number_of_notes = 100
#     number_of_notes = random.randint(20, max_number_of_notes)
#     seq_note = []
#     for x in range(number_of_notes):
#         new_note = Note(random.sample(allowed_pitch, 1)[0], round_down(round(random.uniform(0, 7.75), 2), 0.25), 0.25)
#         # if note is not already on list:
#         if new_note not in seq_note:
#             seq_note.append(new_note)
#     return seq_note
#
#
# def generate(index, pitch, total_number_of_note):
#     """
#
#     :param index:
#     :param pitch:
#     :param total_number_of_note:
#     :return:
#     """
#
#     # Pour 1 mesure : avec une resolution de 64
#     # Ronde = 1
#     # Blanche = 1/2
#     # Noire = 1/4
#     # Croche = 1/8
#     # Double croche = 1/16
#     # Triple croche = 1/32
#     # Quadruple croche = 1/64
#
#     used_credit = 0
#     seq_note = []
#
#     while used_credit < 4:
#         new_note_start = used_credit + index
#         new_note_duration = 4
#         new_note = Note(pitch, new_note_start, new_note_duration)
#         seq_note.append(new_note)
#         used_credit += 4
#     return seq_note
#
#
# def generate_old(index, pitch, total_number_of_note):
#     """
#
#     :param index:
#     :param pitch:
#     :param total_number_of_note:
#     :return:
#     """
#
#     # Pour 1 mesure : avec une resolution de 64
#     # Ronde = 1
#     # Blanche = 1/2
#     # Noire = 1/4
#     # Croche = 1/8
#     # Double croche = 1/16
#     # Triple croche = 1/32
#     # Quadruple croche = 1/64
#
#     resolution_per_bar = 4
#     total_number_of_steps = resolution_per_bar
#     time_credit = total_number_of_steps
#     used_credit = 0
#     seq_note = []
#     # print("duree totale : ", time_credit)
#     while time_credit - used_credit > 0:
#         # print(index)
#         new_note_start = round(used_credit + index, 1)
#         new_note_duration = round(random.uniform(0.1, min(4, time_credit - used_credit)), 1)
#
#         if random.random() > (1 - (1 / total_number_of_note) * 2):
#             # if random.random() > (1 - (1 / total_number_of_note) ):
#             new_note = Note(pitch, new_note_start, new_note_duration)
#             seq_note.append(new_note)
#
#         used_credit = round(used_credit + new_note_duration, 1)
#     # print(seq_note)
#     # print("durée totale:", used_credit)
#     return seq_note
#
#
# class Note:
#     def __init__(self, pitch=None, timestamp=None, duration=None):
#         self.pitch = pitch
#         self.timestamp = timestamp
#         self.duration = duration
#
#     def __eq__(self, obj):
#         return isinstance(obj, Note) and obj.timestamp == self.timestamp and obj.pitch == self.pitch
#
#     def __str__(self):
#         return repr(self)
#
#     def __repr__(self):
#         return f'Pitch : {self.pitch} [t:{self.timestamp} d:{self.duration}]'
#
#
# def convert_to_midi(indiv, file, repertory="output/"):
#     track = 0
#     channel = 0
#     tempo = 60  # In BPM
#     volume = 100  # 0-127, as per the MIDI standard
#
#     my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
#     my_midi.addTempo(track, 0, tempo)
#
#     for bar in indiv.sequence:
#         for note in bar.bit:
#             # print(note)
#             my_midi.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)
#
#     with open(repertory + file, "wb") as output_file:
#         my_midi.writeFile(output_file)
#
#
# def convert_to_midi_seq(seq_note, file, repertory="output/"):
#     track = 0
#     channel = 9
#     tempo = 120  # In BPM
#     volume = 100  # 0-127, as per the MIDI standard
#
#     my_midi = MIDIFile(1)  # One track, defaults to format 1 (tempo track is created automatically)
#     my_midi.addTempo(track, 0, tempo)
#     my_midi.addProgramChange(0, 10, 0, 0)
#     my_midi.tracks[0].addChannelPressure(0, 4, 0)
#
#     for note in seq_note:
#         # print(note)
#         my_midi.addNote(track, channel, note.pitch, note.timestamp, note.duration, volume)
#
#     with open(repertory + file, "wb") as output_file:
#         my_midi.writeFile(output_file)
#
#
# def play_midi_file(file):
#     pygame.mixer.init()
#     pygame.mixer.music.load(file)
#     pygame.mixer.music.play()
#
#     while pygame.mixer.music.get_busy():
#         pass
#
#
# def overlapped_keys(key_to_check, bars):
#     overlapped = []
#     for key in bars:
#         if key_to_check.pitch != key.pitch:
#             if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
#                 overlapped.append(key)
#     return overlapped
#
#
# def check_collision(key_to_check, changed_pitch, bars):
#     for key in bars:
#         if (key_to_check.pitch + changed_pitch) == key.pitch:
#             if key_to_check.timestamp <= key.timestamp <= (key_to_check.timestamp + key_to_check.duration):
#                 return False
#     return True
#
#
# class Bar(Gene):
#
#     def __init__(self, index):
#         super().__init__()
#         # bit == keys
#         self.bit = []
#         self.generate(index * 4)
#
#     def generate(self, index):
#         # automate = automate_bar_generator.create_automate()
#         # while (automate.has_finished() == False):
#         #     self.add_key(automate.next_state())
#         # total_number_of_note = 12
#         # for note in range(total_number_of_note):
#         #    self.add_keys(generate(index, note + 48, total_number_of_note))
#         keys = list(range(1, 13))
#         new_list = random.sample(keys, 3)
#
#         for note in new_list:
#             self.add_keys(generate(index, note + 48, 0))
#
#     def add_keys(self, keys):
#         for key in keys:
#             self.add_key(key)
#
#     def add_key(self, key):
#         self.bit.append(key)
#
#     def mutate(self):
#         for key in self.bit:
#             # probability to switch a key
#             if random.random() > 1 / len(self.bit):
#                 change_pitch = random.randint(-1, 1)
#                 if check_collision(key, change_pitch, self.bit) and 49 <= key.pitch + change_pitch <= 58:
#                     key.pitch += change_pitch
#
#     def __str__(self):
#         return repr(self)
#
#     def __repr__(self):
#         return str(self.bit)
#
#
# class IndividualMusic(Individual):
#     _count = 0
#
#     def __init__(self, parameters):
#         super().__init__(parameters)
#         IndividualMusic._count += 1
#         self.ind = IndividualMusic._count
#         self.sequence = generate_sequence()
#         print(self.sequence)
#         # for index in range(parameters['chromosome size']):
#         #     self.sequence.append(Bar(index))
#
#     def fitness(self):
#
#         convert_to_midi_seq(self.sequence, str(self.ind) + ".mid", "output/")
#         return get_distance("output/" + str(self.ind) + ".mid")
#         # Old fitness :
#         # total_score = 0
#         # number_of_keys = 0
#         # array_score_key = []
#         #
#         # for gene in self.sequence:
#         #     for key in gene.bit:
#         #         number_of_keys += 1
#         #         for overlapped_key in overlapped_keys(key, gene.bit):
#         #             array_score_key.append(12 - abs(overlapped_key.pitch - key.pitch))
#         #
#         # print(array_score_key)
#         # total_score = statistics.mean(array_score_key)
#         #
#         # return total_score / number_of_keys
#
#     def __eq__(self, other):
#         if type(other) != type(self):
#             return False
#         for a, b in zip(self.sequence, other.sequence):
#             if a.bit != b.bit:
#                 return False
#         return True
#
#     def __repr__(self):
#         r = f"I: {self.fitness()}"
#         for g in self.sequence:
#             r += f'\n\t{g.bit}'
#         return r
#
#     def __hash__(self):
#         r = 0
#         for _ in self.sequence:
#             r += random.randint(1, 100)
#         return r
#
#
# # if __name__ == '__main__':
# #     i = IndividualMusic({'chromosome size': 80})
# #     # print(i)
# #     # convert_to_midi(i, "coucou.mid", "../../output/")
# #     print(f'i.fitness() : {i.fitness()}')
#
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('-n', '--name')
# #     parser.add_argument('-p', '--path')
# #     args = parser.parse_args()
# #     # print(args.name)
# #     # print(args.path)
# #     generated_seq = generate_sequence()
# #     convert_to_midi_seq(generated_seq, args.name, args.path)
