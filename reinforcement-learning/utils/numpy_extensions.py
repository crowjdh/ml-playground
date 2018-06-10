import numpy as np


def one_hot(indices, dim):
    if isinstance(indices, int):
        indices = np.asarray(indices).reshape(1)

    i = np.identity(dim, dtype=np.int32)
    return [np.copy(i[idx]) for idx in indices]
