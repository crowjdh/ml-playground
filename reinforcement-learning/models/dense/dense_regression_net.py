from abc import abstractmethod
from typing import *
import tensorflow as tf
import numpy as np
import utils.numpy_extensions as npext
from utils.functions import identity

from models.regression_net import RegressionNet


class DenseRegressionNet(RegressionNet):
    def __init__(self, session, input_dim, output_size, hidden_sizes=(16,),
                 learning_rate=1e-3, use_bias=True, name='main', write_tensor_log=True, log_name_postfix=''):
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        super().__init__(session, learning_rate=learning_rate, use_bias=use_bias, name=name,
                         write_tensor_log=write_tensor_log, log_name_postfix=log_name_postfix)

    def _create_input_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        states = tf.placeholder(tf.float32, [None, self.input_dim], name='states')
        y = tf.placeholder(tf.float32, [None, self.output_size], name='y')

        return states, y

    def _create_network(self, out) -> Tuple[tf.Tensor, tf.Tensor]:
        def dense(inputs, units, desc, activation=None):
            name = 'affine_' + desc
            with tf.name_scope(name):
                dense_layer = tf.layers.Dense(units, activation=activation, use_bias=self.use_bias,
                                              kernel_initializer=self._get_weight_initializer(),
                                              name=name)
                dense_out = dense_layer.apply(inputs)

                tf.summary.histogram('weight_' + desc, dense_layer.kernel, collections=[self.name])
                if self.use_bias:
                    tf.summary.histogram('bias_' + desc, dense_layer.bias, collections=[self.name])

                return dense_out

        with tf.variable_scope(self.name):
            for h_idx, hidden_size in enumerate(self.hidden_sizes):
                out = dense(out, hidden_size, str(h_idx), activation=tf.nn.relu)
            logit = dense(out, self.output_size, str(len(self.hidden_sizes)))

        act = self.activation
        activation_out = act(logit) if callable(act) else logit

        return logit, activation_out

    def predict(self, states):
        states = npext.one_hot(states, self.input_dim, np.float32)
        return super(DenseRegressionNet, self).predict(states)

    def update(self, states, probabilities, feed_dict_processor=identity):
        states = npext.one_hot(states, self.input_dim, np.float32)
        return super(DenseRegressionNet, self).update(states, probabilities, feed_dict_processor=feed_dict_processor)

    @abstractmethod
    def _create_loss_tensor(self) -> tf.Tensor:
        pass

    @abstractmethod
    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        pass

    @abstractmethod
    def log_name(self):
        pass
