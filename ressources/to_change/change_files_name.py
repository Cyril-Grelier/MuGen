import os

files = []

with open('dataset.csv') as d:
    for i, f in enumerate(d.readlines()):
        name = f.split(',')
        name[2] = name[2][:-1]
        files.append((name[1], name[2]))

files.sort(key=lambda x: int(x[1]))

print(files)

for i, f in enumerate(files):
    os.rename("midi_files/" + f[0], "midi_files/" + str(i) + "_" + f[1] + ".mid")
