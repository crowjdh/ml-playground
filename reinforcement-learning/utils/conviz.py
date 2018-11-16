# Blatantly copied from: https://github.com/grishasergei/conviz
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = './.plots'


def plot_conv_weights(weights, name):
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights', name)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    w_min = weights.min()
    w_max = weights.max()

    channel_cnt = weights.shape[2]
    num_filters = weights.shape[3]

    grid_r, grid_c = _get_grid_dim(num_filters)

    row, col = _min_n_max(grid_r, grid_c)
    fig, axes = plt.subplots(row, col)
    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    for channel in range(channel_cnt):
        for l, ax in enumerate(axes.flat):
            img = weights[:, :, channel, l]
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')


def plot_conv_output(conv_img, name):
    plot_dir = os.path.join(PLOT_DIR, 'conv_output', name)
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    w_min = conv_img.min()
    w_max = conv_img.max()

    num_filters = conv_img.shape[3]

    grid_r, grid_c = _get_grid_dim(num_filters)

    row, col = _min_n_max(grid_r, grid_c)
    fig, axes = plt.subplots(row, col)

    axes = axes if isinstance(axes, np.ndarray) else np.array([axes])

    for l, ax in enumerate(axes.flat):
        img = conv_img[0, :, :,  l]
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches='tight')


def _get_grid_dim(x):
    """
    Transforms x into product of two integers
    :param x: int
    :return: two ints
    """
    factors = _prime_powers(x)
    if len(factors) % 2 == 0:
        i = int(len(factors) / 2)
        return factors[i], factors[i - 1]

    i = len(factors) // 2
    return factors[i], factors[i]


def _prime_powers(n):
    """
    Compute the factors of a positive integer
    Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
    :param n: int
    :return: set
    """
    factors = set()
    for x in range(1, int(np.sqrt(n)) + 1):
        if n % x == 0:
            factors.add(int(x))
            factors.add(int(n // x))
    return sorted(factors)


def _min_n_max(lhs, rhs):
    return (lhs, rhs) if lhs < rhs else (rhs, lhs)
