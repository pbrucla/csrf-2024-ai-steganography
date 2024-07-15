from enum import Enum

#Enum to differentiate which dataset to use
class DatasetTypes(Enum):
    CLEAN : 1
    DCT: 2
    FFT: 4
    LSB: 8
    PVD: 16
    SSB4: 32
    SSBN: 64