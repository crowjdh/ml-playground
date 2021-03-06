from typing import Type
import tensorflow as tf

from models.conv.regression_net import ConvRegressionNet
from models.dqn_mixin import DQNMixin


class ConvDQN(ConvRegressionNet, DQNMixin):
    def _init_summaries(self):
        super(ConvDQN, self)._init_summaries()
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

    def _create_loss_tensor(self) -> tf.Tensor:
        return tf.losses.mean_squared_error(self._y, self._activation_out)

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.RMSPropOptimizer
