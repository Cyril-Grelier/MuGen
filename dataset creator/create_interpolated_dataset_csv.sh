#!/bin/bash
#--file_name="generated_"$i".mid"
declare -i number_of_files=200
declare -i number_of_interpolation=20

for i in `seq 1 $number_of_files`;
do
  python generate_random_drums_midi_file.py -n $i"_random"".mid" -p "/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/"
done

echo "generated $number_of_files random midi files"

#generate a 2bar drum loop
drums_rnn_generate \
--config=drum_kit \
--bundle_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/drum_kit_rnn.mag \
--output_dir=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files \
--num_outputs=$number_of_files \
--num_steps=32 \
--primer_drums="[(36,)]" \
--file_name="_generated" \
--is_csv=True

echo "generated $number_of_files good midi files"

#Interpolate them
for i in `seq 1 $number_of_files`;
do
music_vae_generate \
--config=cat-drums_2bar_small \
--checkpoint_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/cat-drums_2bar_small.hikl.ckpt \
--mode=interpolate \
--num_outputs=$number_of_interpolation \
--input_midi_2=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/$i"_generated.mid" \
--input_midi_1=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/$i"_random.mid" \
--output_dir=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/dataset_csv/midi_files/ \
--file_name=$i"_inter_" \
--index=$i \
--is_csv=True
done

