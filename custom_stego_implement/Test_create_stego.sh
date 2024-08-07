#!/bin/bash

mkdir -p data/CustomStego/hamming_codes_binary_Test
mkdir -p data/CustomStego/LSBTest
mkdir -p data/CustomStego/PVDTest

for i in $(find data/CustomStego/cleanTest/ -type f); do
    [ -e "$i" ] || continue
    filename=$(basename "$i")
    poetry run python custom_stego_implement/hamming_codes_random_bit.py "$i" 120000 "data/CustomStego/hamming_codes_binary_Test/$filename"
done

poetry run python custom_stego_implement/LSB_random_bit.py data/CustomStego/cleanTest/ 120000 data/CustomStego/LSBTest/ 
poetry run python custom_stego_implement/PVD_random_bit.py data/CustomStego/cleanTest/ 120000 data/CustomStego/PVDTest/ 

# Wait for all background processes to finish
wait
