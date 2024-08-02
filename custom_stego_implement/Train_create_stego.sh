#!/bin/bash
for i in $(ls data/CustomStego/cleanTrain/); do python customstego_implement/hamming_codes_random_bit.py data/CustomStego/cleanTrain/$i 120000 data/CustomStego/hamming_codes_binary_Train/120k_HAMMING$i; done

python custom_stego_implement/LSB_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/LSBTrain/

python custom_stego_implement/PVD_random_bit.py data/CustomStego/cleanTrain/ 120000 data/CustomStego/PVDTrain/
