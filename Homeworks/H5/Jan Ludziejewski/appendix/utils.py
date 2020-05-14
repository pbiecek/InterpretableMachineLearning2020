def dict_with(dict, key, value):
    new_dict = dict.copy()
    new_dict[key] = value
    return new_dict


def flatten(data):
    return [item for sublist in data for item in sublist]