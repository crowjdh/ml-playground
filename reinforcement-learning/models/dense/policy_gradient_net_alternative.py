import tensorflow as tf
import numpy as np

from models.dense.policy_gradient_net import PGN


'''
Alternative method is from Sung Kim's implementation of policy gradient on CartPole, which can be found here:
- https://github.com/hunkim/ReinforcementZeroToAll/blob/master/09_2_cross_entropy.py
'''


class PGNAlternative(PGN):
    # TODO: Loss makes no sense
    def _init_summaries(self):
        super(PGN, self)._init_summaries()
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

    def _create_loss_tensor(self) -> tf.Tensor:
        self._discounted_rewards = tf.placeholder(tf.float32, [None, 1], name='discounted_rewards')
        log_lik = -self._y * tf.log(self._activation_out)
        log_lik_adv = log_lik * self._discounted_rewards
        return tf.reduce_mean(tf.reduce_sum(log_lik_adv, axis=1))

    def _init_optimizer_tensor(self):
        with tf.name_scope('train'):
            optimizer = self._get_optimizer_type()
            self._loss_tensor = self._create_loss_tensor()
            self._global_step = tf.Variable(0, name='global_step', trainable=False)

            self._optimizer_tensor = optimizer(learning_rate=self.learning_rate).minimize(
                self._loss_tensor, global_step=self._global_step, name='optimizer')

    def perform_policy_gradient_update(self, history):
        y = self._create_fake_label(history)

        def feed_dict_processor(feed_dict):
            feed_dict[self._discounted_rewards] = history.discounted_reward_history

            return feed_dict

        loss = self.update(history.state_history, y, feed_dict_processor)

        return np.mean(loss)
