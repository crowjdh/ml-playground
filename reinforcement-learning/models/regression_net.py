from abc import ABC, abstractmethod
from typing import *
import tensorflow as tf
from utils.functions import identity


class RegressionNet(ABC):
    def __init__(self, session, learning_rate=1e-3, use_bias=True, name='main',
                 write_tensor_log=True, log_name_postfix=''):
        self.session = session
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.name = name

        self.log_name_postfix = log_name_postfix
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
        with tf.name_scope(self.name):
            self._states, self._y = self._create_input_tensors()
            self._logit, self._activation_out = self._create_network(self._states)
            self._init_optimizer_tensor()

    @abstractmethod
    def _create_input_tensors(self) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    @abstractmethod
    def _create_network(self, out) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def dense(self, inputs, units, desc, activation=None):
        name = 'affine_' + desc
        with tf.name_scope(name):
            dense_layer = tf.layers.Dense(units, activation=activation, use_bias=self.use_bias,
                                          kernel_initializer=self._get_weight_initializer(),
                                          name=name)
            dense_out = dense_layer.apply(inputs)

            tf.summary.histogram('weight_' + desc, dense_layer.kernel, collections=[self.name])
            if self.use_bias:
                tf.summary.histogram('bias_' + desc, dense_layer.bias, collections=[self.name])

            return dense_out

    # noinspection PyMethodMayBeStatic
    def _process_input(self, value):
        return value

    def _init_optimizer_tensor(self):
        with tf.name_scope('train'):
            optimizer = self._get_optimizer_type()

            self._loss_tensor = self._create_loss_tensor()
            self._global_step = tf.Variable(0, name='global_step', trainable=False)
            self._optimizer_tensor = optimizer(learning_rate=self.learning_rate).minimize(
                self._loss_tensor, global_step=self._global_step)

    @abstractmethod
    def _create_loss_tensor(self) -> tf.Tensor:
        pass

    @abstractmethod
    def _get_optimizer_type(self) -> Type[tf.train.Optimizer]:
        pass

    @property
    def activation(self):
        return None

    @property
    @abstractmethod
    def log_name(self):
        pass

    @property
    def log_dir_name(self):
        return '.logs/{}_{}'.format(self.log_name, self.log_name_postfix)

    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def _get_weight_initializer(self):
        return tf.contrib.layers.xavier_initializer()

    def _init_summaries(self):
        tf.summary.histogram('input', self._states, collections=[self.name])
        tf.summary.histogram('rewards', self._activation_out, collections=[self.name])
        tf.summary.histogram('y', self._y, collections=[self.name])

    def _prepare_log_dir(self):
        if tf.gfile.Exists(self.log_dir_name):
            tf.gfile.DeleteRecursively(self.log_dir_name)
        tf.gfile.MakeDirs(self.log_dir_name)

    def predict(self, states):
        states = self._process_input(states)

        return self.session.run(self._activation_out, feed_dict={self._states: states})

    def update(self, states, probabilities, feed_dict_processor=identity):
        states = self._process_input(states)

        feed_dict = {self._states: states, self._y: probabilities}
        # TODO: Check identity function works
        feed_dict = feed_dict_processor(feed_dict)

        params = [self._loss_tensor, self._optimizer_tensor, self._global_step]
        if hasattr(self, 'merged_summery') and self.merged_summery is not None:
            params += [self.merged_summery]
            loss, _, step, summary = self.session.run(params, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, global_step=step)
        else:
            loss, _, _ = self.session.run(params, feed_dict=feed_dict)
        return loss
