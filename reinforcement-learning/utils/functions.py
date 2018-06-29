def noop(*params):
    pass


def identity(*param):
    if len(param) == 0:
        return None
    elif len(param) == 1:
        return param[0]
    else:
        return param
