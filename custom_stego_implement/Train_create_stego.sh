#!/bin/bash

mkdir -p data/CustomStego/hamming_codes_binary_Train
mkdir -p data/CustomStego/LSBTrain
mkdir -p data/CustomStego/PVDTrain

for i in $(find data/CustomStego/cleanTrain/ -type f); do
    [ -e "$i" ] || continue
    filename=$(basename "$i")
    poetry run python custom_stego_implement/hamming_codes_random_bit.py "$i" 120000 "data/CustomStego/hamming_codes_binary_Train/$filename"
done 

poetry run python custom_stego_implement/LSB_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/LSBTrain/
poetry run python custom_stego_implement/PVD_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/PVDTrain/

# Wait for all background processes to finish
wait
