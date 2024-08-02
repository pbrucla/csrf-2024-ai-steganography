#!/bin/bash
for i in $(ls data/CustomStego/2kFromcleanVal/); do python customstego_implement/hamming_codes_random_bit.py data/CustomStego/2kFromcleanVal/$i 120000 data/CustomStego/hamming_codes_binary_Val/120k_HAMMING$i; done

python custom_stego_implement/LSB_random_bit.py data/CustomStego/2kFromcleanVal/ 120000 data/CustomStego/LSBVal/

python custom_stego_implement/PVD_random_bit.py data/CustomStego/2kFromcleanVal/ 120000 data/CustomStego/PVDVal/
