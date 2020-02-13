#!/bin/bash
#--file_name="generated_"$i".mid"
declare -i number_of_files=5000

for i in `seq 1 $number_of_files`;
do
  python generate_unquantized_random_drums_midi_file_16_bars.py -n "random_"$i".mid" -p "/Users/Cyril_Musique/Documents/Cours/Dataset_NO_GD/16_bars/0/"
                                                                               
done

#generate a 2bar drum loop
drums_rnn_generate \
--config=drum_kit \
--bundle_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/drum_kit_rnn.mag \
--output_dir=/Users/Cyril_Musique/Documents/Cours/Dataset_NO_GD/16_bars/100 \
--num_outputs=$number_of_files \
--num_steps=256 \
--primer_drums="[(36,)]" \
--file_name="generated_" \

echo "generated $number_of_files random midi files"
