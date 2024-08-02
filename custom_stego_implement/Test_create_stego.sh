#!/bin/bash
for i in $(ls data/CustomStego/1kFromcleanTest/); do python customstego_implement/hamming_codes_random_bit.py data/CustomStego/1kFromcleanTest/$i 120000 data/CustomStego/hamming_codes_binary_Test/120k_HAMMING$i; done

python custom_stego_implement/LSB_random_bit.py data/CustomStego/1kFromcleanTest/ 120000 data/CustomStego/LSBTest/

python custom_stego_implement/PVD_random_bit.py data/CustomStego/2kFromcleanTest/ 120000 data/CustomStego/PVDTest/
