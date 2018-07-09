from typing import Type
import tensorflow as tf
import numpy as np
import random

from models.dense_regression_net import DenseRegressionNet


class DQN(DenseRegressionNet):
    @property
    def log_name(self):
        return 'DQN'

    def _init_summaries(self):
        super(DQN, self)._init_summaries()
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

    def _create_loss_tensor(self) -> tf.Tensor:
        return tf.reduce_mean(tf.square(self._y - self._activation_out))

    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        return tf.train.AdamOptimizer

    def build_copy_variables_from_operation(self, from_dqn):
        operations = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_dqn.name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            operations.append(dest_var.assign(src_var.value()))

        return operations

    @staticmethod
    def replay_train(main_dqn, target_dqn, replay_memory, gamma, minibatch_size=10):
        minibatch = random.sample(replay_memory, minibatch_size)
        minibatch = np.asarray(minibatch)

        states, probabilities = DQN.parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma)
        # states_2, probabilities_2 = DQN.parse_minibatch_naive(minibatch, main_dqn, target_dqn, gamma)

        return main_dqn.update(states, probabilities)

    @staticmethod
    def parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma):
        done_indices = minibatch[:, -1] == True
        undone_indices = minibatch[:, -1] == False

        states = minibatch[:, 0]
        actions = minibatch[:, 1]
        next_states = minibatch[:, 3]

        Qs = main_dqn.predict(states)

        future_rewards = np.max(target_dqn.predict(next_states), axis=1)[undone_indices]
        discounted_future_rewards = gamma * future_rewards

        Qs[done_indices, actions[done_indices]] = minibatch[done_indices, 2]
        Qs[undone_indices, actions[undone_indices]] = minibatch[undone_indices, 2] + discounted_future_rewards

        return states, Qs

    @staticmethod
    def parse_minibatch_naive(minibatch, main_dqn, target_dqn, gamma):
        states = []
        probabilities = []
        for state, action, reward, next_state, done in minibatch:
            Q = main_dqn.predict(state)
            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + gamma * np.max(target_dqn.predict(next_state))

            states.append(state)
            probabilities.append(Q[0])

        return np.array(states), np.array(probabilities)

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
