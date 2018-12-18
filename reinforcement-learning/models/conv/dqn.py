from typing import Type
import tensorflow as tf

from models.conv.regression_net import ConvRegressionNet
from models.dqn_mixin import DQNMixin


class ConvDQN(ConvRegressionNet, DQNMixin):
    def _init_summaries(self):
        super(ConvDQN, self)._init_summaries()
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

    def _create_loss_tensor(self) -> tf.Tensor:
        # Huber loss makes loss smaller, which makes dqn less likely to diverge
        # See: https://en.wikipedia.org/wiki/Huber_loss
        huber_loss = tf.losses.huber_loss(self._y, self._activation_out)
        return tf.reduce_mean(huber_loss)

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.RMSPropOptimizer
