from typing import Type
import tensorflow as tf
import numpy as np

from models.dense.regression_net import DenseRegressionNet


'''
This model is based on Andrej Karpathy's implementation of policy gradient on Pong, converted into Tensorflow.

Source code and detailed explanation can be found here:
- Source code: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
- Explanation: http://karpathy.github.io/2016/05/31/rl/
- Presentation: https://www.youtube.com/watch?v=tqrcjHuNdmQ&t=1403s
'''


class PGN(DenseRegressionNet):
    # TODO: Loss makes no sense
    def _init_summaries(self):
        super(PGN, self)._init_summaries()
        loss = tf.reduce_mean(tf.reduce_sum(self._loss_tensor, axis=1))
        tf.summary.scalar('loss', loss, collections=[self.name])

    @property
    def activation(self):
        return tf.nn.softmax

    def _create_loss_tensor(self) -> tf.Tensor:
        # TODO: Loss becomes zero when reward_sum becomes zero...
        return -tf.log(self._activation_out)

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.AdamOptimizer

    # TODO: What you gotta do to collect gradient with TensorVisualizer?
    def _create_optimizer_tensor(self):
        with tf.name_scope('train'):
            optimizer = self._get_optimizer_type()
            self._loss_tensor = self._create_loss_tensor()
            self._global_step = tf.Variable(0, name='global_step', trainable=False)

            # Manually compute upward gradient rather then computing with fake label(y)
            # See: http://cs231n.github.io/optimization-2/#backprop
            self._gradient_loss = tf.placeholder(tf.float32, name='gradient_loss')
            return optimizer(learning_rate=self.learning_rate).minimize(
                self._loss_tensor, global_step=self._global_step, grad_loss=self._gradient_loss, name='optimizer')

    def _create_fake_label(self, history):
        # Episode    actions  =>  y(fake label)
        #
        #       1    [1,          [[0.0, 1.0, 0.0, 0.0]
        #             3,           [0.0, 0.0, 0.0, 1.0]
        #             2,           [0.0, 0.0, 1.0, 0.0]
        #             1,           [0.0, 1.0, 0.0, 0.0]
        #       2     1,      =>   [0.0, 1.0, 0.0, 0.0]
        #             0,           [1.0, 0.0, 0.0, 0.0]
        #       3     1,           [0.0, 1.0, 0.0, 0.0]
        #             0]           [1.0, 0.0, 0.0, 0.0]]
        y = np.zeros((len(history.state), self.output_size))
        y[np.arange(y.shape[0]), history.action] = 1
        return y

    def perform_policy_gradient_update(self, history):
        y = self._create_fake_label(history)

        def feed_dict_processor(feed_dict):
            # Episode    (y - history.probabilities) * history.discounted_reward
            #
            #       1    ([[0, 1, 0, 0],    [[0.25, 0.29, 0.24, 0.23],)    [[ 1.34],    [[-0.33,  0.96, -0.32, -0.31],
            #            ( [0, 0, 0, 1],     [0.23, 0.24, 0.25, 0.28],)     [ 0.45],     [-0.11, -0.11, -0.11,  0.33],
            #            ( [0, 0, 1, 0],     [0.25, 0.29, 0.24, 0.23],)     [-0.44],     [ 0.11,  0.13, -0.34,  0.10],
            #            ( [0, 1, 0, 0],     [0.24, 0.25, 0.24, 0.27],)     [-1.35],     [ 0.33, -1.01,  0.32,  0.36],
            #       2    ( [0, 1, 0, 0],  -  [0.25, 0.29, 0.24, 0.23],)  *  [ 0.99],  =  [-0.25,  0.71, -0.24, -0.23],
            #            ( [1, 0, 0, 0],     [0.23, 0.24, 0.25, 0.28],)     [-0.99],     [-0.77,  0.24,  0.25,  0.28],
            #       3    ( [0, 1, 0, 0],     [0.25, 0.29, 0.24, 0.23],)     [ 0.99],     [-0.25,  0.71, -0.24, -0.23],
            #            ( [1, 0, 0, 0]]     [0.23, 0.24, 0.25, 0.28]])     [-0.99]]     [-0.77,  0.24,  0.25,  0.28]]
            #
            # Note that * operator does matrix multiplication broadcast, not dot production.
            upward_gradient = (y - history.probabilities) * history.discounted_reward
            feed_dict[self._gradient_loss] = upward_gradient

            return feed_dict

        loss = self.update(history.state, y, feed_dict_processor)

        return np.mean(loss)
