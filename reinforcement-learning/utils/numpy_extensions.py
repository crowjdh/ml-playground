import numpy as np


def one_hot(indices, dim, dtype):
    if isinstance(indices, int):
        indices = np.asarray(indices).reshape(1)

    if not isinstance(indices, np.ndarray):
        indices = np.asarray(indices, dtype=np.int)

    if not np.issubdtype(indices.dtype, np.int64):
        indices = indices.astype(np.int)
    i = np.identity(dim, dtype=dtype)
    return [np.copy(i[idx]) for idx in indices]
