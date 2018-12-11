from utils.functions import identity


def distinct(arr, mapper=identity):
    distinct_array = []
    for element in arr:
        value = mapper(element)
        if value not in distinct_array:
            distinct_array.append(value)

    return distinct_array
