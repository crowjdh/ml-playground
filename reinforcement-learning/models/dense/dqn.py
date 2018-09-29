from typing import Type
import tensorflow as tf
import numpy as np

from models.dense.regression_net import DenseRegressionNet
from models.dqn_mixin import DQNMixin


class DenseDQN(DenseRegressionNet, DQNMixin):
    @property
    def log_name(self):
        return 'DenseDQN'

    def _init_summaries(self):
        super(DenseDQN, self)._init_summaries()
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

    def _create_loss_tensor(self) -> tf.Tensor:
        return tf.reduce_mean(tf.square(self._y - self._activation_out))

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.AdamOptimizer

    @staticmethod
    def create_dummy_minibatch():
        states = np.arange(16, dtype=np.int32).reshape(1, 16).T
        actions = np.array([0, 1, 2, 3] * 4).reshape(1, 16).T
        rewards = np.zeros((1, 16)).T
        next_state = np.arange(5, 5 + 16).reshape(1, 16).T % 16
        done = np.repeat(False, 16).reshape((1, 16)).T

        rewards[[0, 8]] = 1
        done[[1, 9]] = True

        return np.hstack([states, actions, rewards, next_state, done]).astype(int).tolist()
