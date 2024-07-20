from enum_utils import IntEnum

#Enum to differentiate which dataset to use
class DatasetTypes(IntEnum):
    CLEAN = 1
    DCT = 2
    FFT = 4
    LSB = 8
    PVD = 16
    SSB4 = 32
    SSBN = 64