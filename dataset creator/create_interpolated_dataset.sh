#!/bin/bash
#--file_name="generated_"$i".mid"
declare -i number_of_files=99

for i in `seq 1 $number_of_files`;
do
  python generate_random_drums_midi_file.py -n "random_"$i".mid" -p "/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/"
done

#generate a 2bar drum loop
drums_rnn_generate \
--config=drum_kit \
--bundle_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/drum_kit_rnn.mag \
--output_dir=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100 \
--num_outputs=$number_of_files \
--num_steps=32 \
--primer_drums="[(36,)]" \
--file_name="generated_"

echo "generated $number_of_files random midi files"

#Interpolate them
for i in `seq 1 $number_of_files`;
do
music_vae_generate \
--config=cat-drums_2bar_small \
--checkpoint_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/cat-drums_2bar_small.hikl.ckpt \
--mode=interpolate \
--num_outputs=$number_of_files \
--input_midi_2=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/100/generated_$i.mid \
--input_midi_1=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/0/random_$i.mid \
--output_dir=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/DATASET/ \
--file_name="inter_" \
--index=$i
done
