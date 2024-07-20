# import DatasetTypes enum from config 
from config import DatasetTypes

#since the datset argument takes in a list of strings, this is used to convert that list back to integers for processing later
def enum_names_to_values(names):
    values = []
    for name in names:
        member = DatasetTypes[name]
        values.append(member.value)
    return values