#!/bin/bash

mkdir -p data/CustomStego/hamming_codes_binary_Val
mkdir -p data/CustomStego/LSBVal
mkdir -p data/CustomStego/PVDVal

for i in $(find data/CustomStego/cleanVal/ -type f); do
    [ -e "$i" ] || continue
    filename=$(basename "$i")
    poetry run python custom_stego_implement/hamming_codes_random_bit.py "$i" 120000 "data/CustomStego/hamming_codes_binary_Val/$filename"
done

poetry run python custom_stego_implement/LSB_random_bit.py data/CustomStego/cleanVal/ 120000 data/CustomStego/LSBVal/ 
poetry run python custom_stego_implement/PVD_random_bit.py data/CustomStego/cleanVal/ 120000 data/CustomStego/PVDVal/

wait
