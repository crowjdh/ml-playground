from abc import abstractmethod
from typing import *
from collections import Iterable
from numbers import Complex

import tensorflow as tf
import numpy as np

from models.regression_net import RegressionNet
from utils import image_util


class ConvRegressionNet(RegressionNet):
    PADDING_CANDIDATES = ['SAME', 'VALID']

    """
      Args:
        input_shape: [C, W, H]
        filter_shapes: [[F_N, F_C, WW, HH], ...]
        strides: Array which contains stride of each filter. Should be:
            len(strides) == len(filter_shapes)
        paddings: Array which contains padding algorithms for each convolution.. Should be:
            len(paddings) == len(filter_shapes)
    """
    def __init__(self, session, env_id, input_shape, filter_shapes, strides, paddings, hidden_sizes, output_size,
                 learning_rate=1e-3, use_bias=True, name='main', write_tensor_log=True, visualize=True,
                 resize_ratio=None):
        self.input_shape = list(input_shape)
        self.filter_shapes = filter_shapes
        self.strides = strides
        self.paddings = paddings
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.resize_ratio = resize_ratio

        self.input_shape[:2] = self.resize(self.input_shape[:2])

        super().__init__(session, env_id, learning_rate=learning_rate, use_bias=use_bias, name=name,
                         write_tensor_log=write_tensor_log, visualize=visualize)

    def resize(self, obj):
        if self.resize_ratio is None:
            return obj
        if isinstance(obj, Iterable):
            return [int(l * self.resize_ratio) for l in obj]
        elif isinstance(obj, Complex):
            return int(obj * self.resize_ratio)
        else:
            raise ValueError("Cannot operate resize to {} type".format(type(obj)))

    def calc_output_shape(self):
        # noinspection PyShadowingNames
        def calc_pad_output_size(padding, in_size, filter_size, stride):
            if padding == 'SAME':
                return in_size

            from math import ceil
            return ceil(float(in_size - filter_size + 1) / float(stride))

        assert len(self.input_shape) == 4
        assert len(self.filter_shapes.shape) == 2  # self.filter_shapes.shape: (filter_count, 4)
        assert self.filter_shapes.shape[1] == 4
        assert len(self.strides) == len(self.paddings) == len(self.filter_shapes)

        in_shape = self.input_shape
        _, c, w, h = in_shape

        for idx in range(len(self.filter_shapes)):
            filter_shape = self.filter_shapes[idx]
            padding = self.paddings[idx]
            stride = self.strides[idx]
            f_n, f_c, ww, hh = filter_shape

            assert f_c == c
            assert padding in ConvRegressionNet.PADDING_CANDIDATES

            c = f_n
            w = calc_pad_output_size(padding, w, ww, stride)
            h = calc_pad_output_size(padding, h, hh, stride)

            in_shape = (_, c, w, h)

        return in_shape

    def _create_input_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        states = tf.placeholder(tf.float32, [None] + list(self.input_shape), name='states')
        y = tf.placeholder(tf.float32, [None, self.output_size], name='y')

        return states, y

    def summarise_conv_layer(self, conv_layer, desc):
        channels = conv_layer.kernel.shape[2]
        channel_start = 0
        image_idx = 0
        while channel_start < channels:
            channel_end = channel_start + 4
            if channel_end > channels:
                avail_channels = channels % 4
                if avail_channels == 0:
                    avail_channels = 4
                elif avail_channels == 2:
                    avail_channels = 1
                channel_end = channel_start + avail_channels
            k = conv_layer.kernel[:, :, channel_start:channel_end, image_idx:image_idx+1]
            f = tf.transpose(k, [3, 0, 1, 2])
            tf.summary.image('filter_{}_ch_{}-{}'.format(desc, channel_start, channel_end), f, collections=[self.name])
            channel_start = channel_end

    def conv(self, inputs, idx, desc, activation=None):
        name = 'conv_' + desc
        with tf.name_scope(name):
            filter_shape, stride, padding = self.filter_shapes[idx], self.strides[idx], self.paddings[idx]
            f_n, f_c, ww, hh = filter_shape

            conv_layer = tf.layers.Conv2D(f_n, (hh, ww), stride,
                                          name=name, padding=padding, activation=activation,
                                          use_bias=self.use_bias,
                                          kernel_initializer=self._get_weight_initializer(),
                                          bias_initializer=self._get_bias_initializer())
            conv_out = conv_layer.apply(inputs)

            tf.summary.histogram('weight_' + desc, conv_layer.kernel, collections=[self.name])
            if self.use_bias:
                tf.summary.histogram('bias_' + desc, conv_layer.bias, collections=[self.name])

            return conv_layer, conv_out

    def _create_network(self, out) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.variable_scope(self.name):
            for f_idx in range(len(self.filter_shapes)):
                layer, out = self.conv(out, f_idx, str(f_idx), activation=tf.nn.relu)

                self.collect_conv_layer_tensors(layer, out)
            out = tf.reshape(out, (-1, np.prod(out.shape[1:])))
            for h_idx, hidden_size in enumerate(self.hidden_sizes):
                layer, out = self.dense(out, hidden_size, str(h_idx), activation=tf.nn.relu)
                self.collect_dense_layer_tensors(layer, out)
            layer, logit = self.dense(out, self.output_size, str(len(self.hidden_sizes)))

        act = self.activation
        activation_out = act(logit) if callable(act) else logit
        self.collect_dense_layer_tensors(layer, activation_out)

        return logit, activation_out

    def _process_input(self, values):
        if len(values) == 0:
            return None
        if len(values.shape) == 3:
            values = values[np.newaxis]
        if self.resize_ratio is not None:
            values = image_util.resize(values, self.input_shape[:2])
        values = (values - values.mean()) / values.std()

        return values

    @abstractmethod
    def _create_loss_tensor(self) -> tf.Tensor:
        pass

    @abstractmethod
    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        pass
