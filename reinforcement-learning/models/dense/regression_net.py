from abc import abstractmethod
from typing import *
import tensorflow as tf
import numpy as np
import utils.numpy_extensions as npext

from models.regression_net import RegressionNet


class DenseRegressionNet(RegressionNet):
    def __init__(self, session, env_id, input_dim, output_size, hidden_sizes=(16,),
                 learning_rate=1e-3, use_bias=True, name='main', write_tensor_log=True):
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        super().__init__(session, env_id, learning_rate=learning_rate, use_bias=use_bias, name=name,
                         write_tensor_log=write_tensor_log)

    def _create_input_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        states = tf.placeholder(tf.float32, [None, self.input_dim], name='states')
        y = tf.placeholder(tf.float32, [None, self.output_size], name='y')

        return states, y

    def _create_network(self, out) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.variable_scope(self.name):
            for h_idx, hidden_size in enumerate(self.hidden_sizes):
                out = self.dense(out, hidden_size, str(h_idx), activation=tf.nn.relu)
            logit = self.dense(out, self.output_size, str(len(self.hidden_sizes)))

        act = self.activation
        activation_out = act(logit) if callable(act) else logit

        return logit, activation_out

    def _process_input(self, values):
        return npext.one_hot(values, self.input_dim, np.float32)

    @abstractmethod
    def _create_loss_tensor(self) -> tf.Tensor:
        pass

    @abstractmethod
    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        pass
