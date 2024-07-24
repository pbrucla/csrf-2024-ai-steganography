from enum import IntEnum


# Enum to differentiate which dataset to use
class DatasetTypes(IntEnum):
    CLEAN = 1
    DCT = 2
    FFT = 4
    LSB = 8
    PVD = 16
    SSB4 = 32
    SSBN = 64


# since the datset argument takes in a list of strings, this is used to convert that list back to integers for processing later
def enum_names_to_values(names):
    values = []
    for name in names:
        member = DatasetTypes[name]
        values.append(member.value)
    return values
