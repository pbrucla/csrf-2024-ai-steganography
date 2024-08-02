#!/bin/bash
for i in $(ls data/CustomStego/cleanTrain/); do poetry run python custom_stego_implement/hamming_codes_random_bit.py data/CustomStego/cleanTrain/$i 120000 data/CustomStego/hamming_codes_binary_Train/120k_HAMMING$i; done &

poetry run python custom_stego_implement/LSB_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/LSBTrain/ &

poetry run python custom_stego_implement/PVD_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/PVDTrain/
