import music21

# fp = '0.mid'
# mf = music21.midi.MidiFile()
# mf.open(str(fp))
# mf.read()
# mf.close()
# len(mf.tracks)
#
# s = music21.midi.translate.midiFileToStream(mf)
# print(s)
#
# print(len(s.flat.notesAndRests))
#
# s.show()

s = music21.converter.parse('0.mid')

print(s)
