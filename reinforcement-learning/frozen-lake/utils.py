import numpy as np
import random as pr


def rand_argmax(arr):
    maxima = np.amax(arr)
    indices = np.nonzero(arr == maxima)[0]
    return pr.choice(indices)
