import numpy as np


def resize(images, resize_to):
    """
    Resize image to

    Parameters
    ----------
    images : np.ndarray
        Numpy array of shape either (N, W, H, C) or (W, H, C).
    resize_to : array_like
        Tuple or array of shape (resized_W, resized_H).

    Returns
    -------
    res : ndarray
        Resized image.
    """
    dim_count = len(images.shape)
    assert dim_count == 3 or dim_count == 4
    assert len(resize_to) == 2

    prepend_dim = dim_count == 3

    if prepend_dim:
        images = np.expand_dims(images, axis=0)
        dim_count = 4

    width_height_axes = tuple(range(dim_count))[1:-1]
    batch_size, channel_count = images.shape[0], images.shape[3]

    in_width, in_height = images.shape[1], images.shape[2]
    out_width, out_height = resize_to

    width_sample_freq = int(in_width / out_width)
    height_sample_freq = int(in_height / out_height)
    output = np.empty((batch_size, out_width, out_height, channel_count))
    for w_idx in range(in_width):
        for h_idx in range(in_height):
            if w_idx % width_sample_freq == 0 and h_idx % height_sample_freq == 0:
                out_w_idx, out_h_idx = int(w_idx / width_sample_freq), int(h_idx / height_sample_freq)
                fraction = images[:, w_idx:w_idx + width_sample_freq, h_idx:h_idx + height_sample_freq]
                output[:, out_w_idx, out_h_idx] = fraction.mean(axis=width_height_axes)

    if prepend_dim:
        output = np.squeeze(output, axis=0)

    return output
