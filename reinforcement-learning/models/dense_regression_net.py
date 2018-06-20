from abc import ABC, abstractmethod
from typing import Type
from datetime import datetime
import tensorflow as tf
import numpy as np
import utils.numpy_extensions as npext


class DenseRegressionNet(ABC):
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
        optimizer = self._get_optimizer_type()

        self._loss_tensor = self._create_loss_tensor()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self._optimizer_tensor = optimizer(learning_rate=self.learning_rate).minimize(self._loss_tensor, global_step=self.global_step)

    @abstractmethod
    def _create_loss_tensor(self) -> tf.Tensor:
        pass

    @abstractmethod
    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        pass

    def _init_summaries(self):
        tf.summary.histogram('input', self._states, collections=[self.name])
        tf.summary.histogram('rewards', self._rewards, collections=[self.name])
        tf.summary.histogram('y', self._y, collections=[self.name])
        tf.summary.scalar('loss', self._loss_tensor, collections=[self.name])

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
            summary, loss, _, step = self.session.run([self.merged_summery, self._loss_tensor, self._optimizer_tensor, self.global_step],
                                                      feed_dict={self._states: states, self._y: probabilities})
            self.train_writer.add_summary(summary, global_step=step)
        else:
            loss, _ = self.session.run([self._loss_tensor, self._optimizer_tensor],
                                       feed_dict={self._states: states, self._y: probabilities})
        return loss
