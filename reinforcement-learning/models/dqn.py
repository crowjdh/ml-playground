import random
from datetime import datetime
from collections import Iterable
import tensorflow as tf
import numpy as np
import utils.numpy_extensions as npext


class DQN:
    def __init__(self, session, input_dim, output_size, hidden_sizes=[10], learning_rate=1e-3, name='main', Q_formatter=None, write_tensor_log=True):
        self.session = session
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.name = name
        self.Q_formatter = Q_formatter

        log_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.log_file_name = '.logs/{}.txt'.format(log_name)
        self.log_dir_name = '.logs/{}'.format(log_name)
        self.write_tensor_log = write_tensor_log

        self._build_graph()

    def __enter__(self):
        if self.write_tensor_log:
            self._prepare_log_dir()
            self.merged_summery = tf.summary.merge_all(key=self.name)
            self.train_writer = tf.summary.FileWriter(self.log_dir_name, self.session.graph)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.write_tensor_log:
            self.train_writer.close()

    def _build_graph(self):
        def variable_summaries(var):
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean, collections=[self.name])
                with tf.name_scope('stddev'):
                    m = tf.reduce_mean(tf.square(var - mean))
                    stddev = tf.sqrt(m)
                tf.summary.scalar('stddev', stddev, collections=[self.name])
                tf.summary.scalar('max', tf.reduce_max(var), collections=[self.name])
                tf.summary.scalar('min', tf.reduce_min(var), collections=[self.name])
                tf.summary.histogram('histogram', var, collections=[self.name])

        def weight_variable(key, shape):
            # return tf.get_variable(key, shape=shape, initializer=tf.zeros_initializer())
            return tf.get_variable(key, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

        def bias_variable(key, shape):
            # return tf.get_variable(key, shape=shape, initializer=tf.zeros_initializer())
            return tf.get_variable(key, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            with tf.name_scope(layer_name):
                with tf.name_scope('weights'):
                    weights = weight_variable(layer_name + '/weight', (input_dim, output_dim))
                with tf.name_scope('biases'):
                    biases = bias_variable(layer_name + '/bias', [output_dim])
                with tf.name_scope('Wx'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    # tf.summary.histogram('Wx', preactivate, collections=[self.name])
                activations = act(preactivate, name='activation')
                return activations

        with tf.variable_scope(self.name):
            self._states = tf.placeholder(tf.int32, [None, self.input_dim], name='states')
            out = tf.cast(self._states, tf.float32)
            # self.aa = out = tf.layers.batch_normalization(out)
            tf.summary.histogram('input', out, collections=[self.name])

            for h_idx, hidden_size in enumerate(self.hidden_sizes):
                # W = weight_variable('W' + str(h_idx), (int(out.shape[-1]), hidden_size))
                # out = tf.matmul(out, W)
                # out = tf.layers.batch_normalization(out)
                out = nn_layer(out, int(out.shape[-1]), hidden_size, 'layer' + str(h_idx), act=tf.nn.tanh)
                tf.summary.histogram('Wx_activations', out, collections=[self.name])
                # variable_summaries(out)
                # W, out = DQN.affine(out, 'W' + str(h_idx), out.shape[-1], hidden_size)
                # out = tf.nn.tanh(out, name='activation')

            # out = tf.layers.batch_normalization(out)

            # W = weight_variable('W_last', (out.shape[-1], self.output_size))
            # self._probabilities = tf.matmul(out, W)
            # out = tf.layers.batch_normalization(out)
            self._rewards = nn_layer(out, int(out.shape[-1]), self.output_size, 'layer' + str(len(self.hidden_sizes)), act=tf.identity)
            tf.summary.histogram('rewards', self._rewards, collections=[self.name])
            # variable_summaries(self._probabilities)
            # W, out = DQN.affine(out, 'W_last', out.shape[-1], self.output_size)

            self._y = tf.placeholder(tf.float32, [None, self.output_size], name='y')
            tf.summary.histogram('y', self._y, collections=[self.name])

            self._L2_loss = tf.reduce_mean(tf.square(self._y - self._rewards))
            tf.summary.scalar('loss', self._L2_loss, collections=[self.name])
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)\
                .minimize(self._L2_loss, global_step=self.global_step)

    def _prepare_log_dir(self):
        if tf.gfile.Exists(self.log_dir_name):
            tf.gfile.DeleteRecursively(self.log_dir_name)
        tf.gfile.MakeDirs(self.log_dir_name)

    def get_Q(self):
        # TODO: Resolve dependency
        return self.predict(range(16))

    def predict(self, states):
        states = DQN.reshape_states(states)
        # states = npext.one_hot(states, self.input_dim)
        return self.session.run(self._rewards, feed_dict={self._states: states})

    def update(self, states, probabilities):
        states = DQN.reshape_states(states)
        # states = npext.one_hot(states, self.input_dim)
        # TODO: Check whether we need L2 loss
        if self.merged_summery is not None:
            summary, loss, _, step = self.session.run([self.merged_summery, self._L2_loss, self._optimizer, self.global_step],
                                                feed_dict={self._states: states, self._y: probabilities})
            self.train_writer.add_summary(summary, global_step=step)
        else:
            loss, _ = self.session.run([self._L2_loss, self._optimizer],
                                       feed_dict={self._states: states, self._y: probabilities})
        return loss

    def build_copy_variables_from_operation(self, from_dqn):
        operations = []
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_dqn.name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        for src_var, dest_var in zip(src_vars, dest_vars):
            operations.append(dest_var.assign(src_var.value()))

        return operations

    @staticmethod
    def reshape_states(states):
        return np.array(states).reshape(1, -1).T

    @staticmethod
    def replay_train(main_dqn, target_dqn, replay_memory, gamma, minibatch_size=10):
        minibatch_size = min(len(replay_memory), minibatch_size)
        minibatch = random.sample(replay_memory, minibatch_size)

        minibatch = np.asarray(minibatch)

        states, probabilities = DQN.parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma)
        # states_2, probabilities_2 = DQN.parse_minibatch_naive(minibatch, main_dqn, target_dqn, gamma)

        main_dqn.log('\nTarget DQN:')
        main_dqn.log(target_dqn.Q_formatter(target_dqn.get_Q()))

        return main_dqn.update(states, probabilities)

    @staticmethod
    def parse_minibatch_vectorized(minibatch, main_dqn, target_dqn, gamma):
        done_indices = minibatch[:, -1] == True
        undone_indices = minibatch[:, -1] == False

        states = minibatch[:, 0]
        actions = minibatch[:, 1]
        next_states = minibatch[:, 3]

        Qs = main_dqn.predict(states)

        # main_dqn.log('\nBatch:')
        # main_dqn.log(minibatch)

        future_rewards = np.max(target_dqn.predict(next_states[undone_indices]), axis=1)
        discounted_future_rewards = gamma * future_rewards

        Qs[done_indices, actions[done_indices]] = minibatch[done_indices, 2]
        Qs[undone_indices, actions[undone_indices]] = minibatch[undone_indices, 2] + discounted_future_rewards

        return states, Qs

    @staticmethod
    def parse_minibatch_naive(minibatch, main_dqn, target_dqn, gamma):
        states = []
        probabilities = []
        # Straightforward algorithm
        for state, action, reward, next_state, done in minibatch:
            Q = main_dqn.predict(state)
            if done:
                Q[0, action] = reward
            else:
                Q[0, action] = reward + gamma * np.max(target_dqn.predict(next_state))

            states.append(state)
            probabilities.append(Q[0])

        return np.array(states), np.array(probabilities)

    def log(self, lines):
        if isinstance(lines, str) or not isinstance(lines, Iterable):
            lines = [lines]
        with open(self.log_file_name, 'a') as log_file:
            for line in lines:
                log_file.write(str(line) + '\n')
