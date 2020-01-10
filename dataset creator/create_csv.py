import csv
from os import listdir
from os.path import isfile, join
columns = ['Name', 'File', 'Score']

path = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/'
filename = '/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/dataset.csv'
#Clear the file
f = open(filename, "w+")
f.close()

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)

with open(filename, 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(columns)
    for file in onlyfiles:
        if "DS" not in file:
            nomid = file.replace('.mid', '')
            score = 0
            if "random" in file:
                score = 0
            elif "generate" in file:
                score = 100
            else:
                print(file)
                score = int(nomid[-2:].replace('_',''))
                score *=5


            row = [nomid,file ,score]

            writer.writerow(row)

csvFile.close()