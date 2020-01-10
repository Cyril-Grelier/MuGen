#!/bin/bash
#--file_name="generated_"$i".mid"
declare -i number_of_files=10


for i in `seq 1 $number_of_files`;
do
  python generate_unquantized_random_drums_midi_file.py -n "random_"$i".mid" -p "/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_0/"
                                                                               
done


#generate a 2bar drum loop
drums_rnn_generate \
--config=drum_kit \
--bundle_file=/Users/Cyril_Musique/Documents/Cours/M2/PROJETALGOMUSIQUE/interpolate/drum_kit_rnn.mag \
--output_dir=/Users/Cyril_Musique/Documents/Cours/M2/MuGen/datasets/quantized_rythm_dataset_v2_temperature/to_compare_100 \
--num_outputs=$number_of_files \
--num_steps=32 \
--primer_drums="[(36,)]" \
--file_name="generated_" \

echo "generated $number_of_files random midi files"
