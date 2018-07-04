from typing import Type
import tensorflow as tf
import numpy as np

from models.dense_regression_net import DenseRegressionNet


'''
This model is based on Andrej Karpathy's implementation of policy gradient on Pong, converted into Tensorflow.

Source code and detailed explanation can be found here:
- Source code: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
- Explanation: http://karpathy.github.io/2016/05/31/rl/
'''


class PGN(DenseRegressionNet):
    """
    Alternative method is from Sung Kim's implementation of policy gradient on CartPole, which can be found here:
    - https://github.com/hunkim/ReinforcementZeroToAll/blob/master/09_2_cross_entropy.py
    """
    # TODO: Consider subclassing for alternative method
    def __init__(self, *args, use_alternative_optimizer_method=False, **kwargs):
        self.use_alternative_optimizer_method = use_alternative_optimizer_method
        super(PGN, self).__init__(*args, **kwargs)

    # TODO: Loss makes no sense
    def _init_summaries(self):
        super(PGN, self)._init_summaries()
        loss = tf.reduce_mean(tf.reduce_sum(self._loss_tensor, axis=1))
        tf.summary.scalar('loss', loss, collections=[self.name])

    @property
    def activation(self):
        return tf.nn.softmax

    def _create_loss_tensor(self) -> tf.Tensor:
        if self.use_alternative_optimizer_method:
            self._discounted_rewards = tf.placeholder(tf.float32, [None, 1], name='discounted_rewards')
            log_lik = -self._y * tf.log(self._activation_out)
            log_lik_adv = log_lik * self._discounted_rewards
            return tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

        # TODO: Loss becomes zero when reward_sum becomes zero...
        return -tf.log(self._activation_out)

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.AdamOptimizer

    def _init_optimizer_tensor(self):
        with tf.name_scope('train'):
            optimizer = self._get_optimizer_type()

            self._loss_tensor = self._create_loss_tensor()
            self._global_step = tf.Variable(0, name='global_step', trainable=False)

            if self.use_alternative_optimizer_method:
                self._optimizer_tensor = optimizer(learning_rate=self.learning_rate).minimize(
                    self._loss_tensor, global_step=self._global_step, name='optimizer')
            else:
                self._gradient_loss = tf.placeholder(tf.float32, name='gradient_loss')
                self._optimizer_tensor = optimizer(learning_rate=self.learning_rate).minimize(
                    self._loss_tensor, global_step=self._global_step, grad_loss=self._gradient_loss, name='optimizer')

    def perform_policy_gradient_update(self, history):
        y = np.zeros((len(history.state_history), self.output_size))
        y[np.arange(y.shape[0]), history.action_history] = 1

        manual_loss = (y - history.probabilities_history) * history.discounted_reward_history

        def feed_dict_processor(feed_dict):
            if self.use_alternative_optimizer_method:
                feed_dict[self._discounted_rewards] = history.discounted_reward_history
            else:
                feed_dict[self._gradient_loss] = manual_loss

            return feed_dict

        loss = self.update(history.state_history, y, feed_dict_processor)

        return np.mean(loss)
