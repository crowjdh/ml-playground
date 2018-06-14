import random
from datetime import datetime
import tensorflow as tf
import numpy as np
import utils.numpy_extensions as npext


class DQN:
    def __init__(self, session, input_dim, output_size, hidden_sizes=(16,),
                 learning_rate=1e-3, name='main', write_tensor_log=True):
        self.session = session
        self.input_dim = input_dim
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.name = name

        self.log_name = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.log_dir_name = '.logs/{}'.format(self.log_name)
        self.write_tensor_log = write_tensor_log

        self._build_graph()
        self._init_summaries()

    def __enter__(self):
        if self.write_tensor_log:
            self._prepare_log_dir()
            self.merged_summery = tf.summary.merge_all(key=self.name)
            self.train_writer = tf.summary.FileWriter(self.log_dir_name, self.session.graph)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.write_tensor_log:
            self.train_writer.close()

    def _build_graph(self):
        with tf.variable_scope(self.name):
            self._init_input_tensors()
            self._init_network(self._states)
            self._init_optimizer_tensor()

    def _init_input_tensors(self):
        self._states = tf.placeholder(tf.float32, [None, self.input_dim], name='states')
        self._y = tf.placeholder(tf.float32, [None, self.output_size], name='y')

    def _init_network(self, out):
        def dense(inputs, units, desc, activation=None):
            dense_layer = tf.layers.Dense(units, activation=activation)
            res = dense_layer.apply(inputs)

            tf.summary.histogram('weight_' + desc, dense_layer.kernel, collections=[self.name])
            tf.summary.histogram('bias_' + desc, dense_layer.bias, collections=[self.name])

            return res

        for h_idx, hidden_size in enumerate(self.hidden_sizes):
            out = dense(out, hidden_size, str(h_idx), activation=tf.nn.relu)
        self._rewards = dense(out, self.output_size, 'last')

    def _init_optimizer_tensor(self):
        self._L2_loss = tf.reduce_mean(tf.square(self._y - self._rewards))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate) \
            .minimize(self._L2_loss, global_step=self.global_step)

    def _init_summaries(self):
        tf.summary.histogram('input', self._states, collections=[self.name])
        tf.summary.histogram('rewards', self._rewards, collections=[self.name])
        tf.summary.histogram('y', self._y, collections=[self.name])
        tf.summary.scalar('loss', self._L2_loss, collections=[self.name])

    def _prepare_log_dir(self):
        if tf.gfile.Exists(self.log_dir_name):
            tf.gfile.DeleteRecursively(self.log_dir_name)
        tf.gfile.MakeDirs(self.log_dir_name)

    def predict(self, states):
        states = npext.one_hot(states, self.input_dim, np.float32)
        return self.session.run(self._rewards, feed_dict={self._states: states})

    def update(self, states, probabilities):
        states = npext.one_hot(states, self.input_dim, np.float32)
        if hasattr(self, 'merged_summery') and self.merged_summery is not None:
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
