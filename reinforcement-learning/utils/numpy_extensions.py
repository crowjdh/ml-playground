import numpy as np


def one_hot(indices, dim, dtype):
    if isinstance(indices, int):
        indices = np.asarray(indices).reshape(1)

    i = np.identity(dim, dtype=dtype)
    return [np.copy(i[idx]) for idx in indices]
