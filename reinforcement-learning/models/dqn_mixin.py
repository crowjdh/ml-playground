import tensorflow as tf
import numpy as np
import random


# noinspection PyUnresolvedReferences
class DQNMixin:
    def build_copy_variables_from_operation(self, from_dqn):
        operations = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_dqn.name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            operations.append(dest_var.assign(src_var.value()))

        return operations

    @classmethod
    def replay_train(cls, main_dqn, target_dqn, replay_memory, gamma, minibatch_size=10):
        minibatch = random.sample(replay_memory, minibatch_size)
        minibatch = np.asarray(minibatch)

        states, probabilities = cls.parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma)
        # states_2, probabilities_2 = DQN.parse_minibatch_naive(minibatch, main_dqn, target_dqn, gamma)

        return main_dqn.update(states, probabilities)

    @staticmethod
    def parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma):
        def unravel(values):
            shape = [values.shape[0]] + list(values[0].shape)
            new_values = np.empty(shape, np.int)
            for i in range(len(values)):
                value = values[i]
                new_values[i] = value

            return new_values

        done_indices = minibatch[:, -1] == True
        undone_indices = minibatch[:, -1] == False

        states = unravel(minibatch[:, 0])
        actions = np.asarray(minibatch[:, 1], dtype=np.int)
        next_states = unravel(minibatch[:, 3])

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
