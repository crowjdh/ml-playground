from abc import ABC, abstractmethod
from typing import *
import tensorflow as tf
from utils.functions import identity
from models.tensor_visualizer import TensorVisualizer
from utils.path import cache_dir_path


class RegressionNet(ABC):
    def __init__(self, session, env_id, learning_rate=1e-3, use_bias=True, name='main',
                 write_tensor_log=True, visualize=True):
        self.session = session
        self.env_id = env_id
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.name = name

        self.write_tensor_log = write_tensor_log
        self.visualize = visualize

        self._build_graph()
        self._init_summaries()

    def __enter__(self):
        if self.write_tensor_log:
            self.merged_summery = tf.summary.merge_all(key=self.name)
            self.train_writer = tf.summary.FileWriter(self.log_dir_path, self.session.graph)

    def __exit__(self, exception_type, exception_value, traceback):
        if self.write_tensor_log:
            self.train_writer.close()

    def _build_graph(self):
        with tf.name_scope(self.name):
            self._states, self._y = self._create_input_tensors()
            self._logit, self._activation_out = self._create_network(self._states)
            self._optimizer_tensor = self._create_optimizer_tensor()

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
                                          bias_initializer=self._get_bias_initializer(),
                                          name=name)
            dense_out = dense_layer.apply(inputs)

            tf.summary.histogram('weight_' + desc, dense_layer.kernel, collections=[self.name])
            if self.use_bias:
                tf.summary.histogram('bias_' + desc, dense_layer.bias, collections=[self.name])

            return dense_layer, dense_out

    def collect_dense_layer_tensors(self, layer, activation):
        if self.visualize:
            TensorVisualizer.instance.add_dense_layer(layer, activation)

    def collect_conv_layer_tensors(self, layer, activation):
        if self.visualize:
            TensorVisualizer.instance.add_conv_layer(layer, activation)

    # noinspection PyMethodMayBeStatic
    def _process_input(self, values):
        return values

    def _create_optimizer_tensor(self):
        with tf.name_scope('train'):
            optimizer = self._get_optimizer_type()

            self._loss_tensor = self._create_loss_tensor()
            self._global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer_layer = optimizer(learning_rate=self.learning_rate)
            grads_and_vars = optimizer_layer.compute_gradients(self._loss_tensor)
            if self.visualize:
                TensorVisualizer.instance.collect_gradients(grads_and_vars)
            return optimizer_layer.apply_gradients(grads_and_vars, global_step=self._global_step)

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
    def id(self):
        return '{}_{}'.format(self.env_id, self.__class__.__name__)

    @property
    def log_dir_path(self):
        return cache_dir_path(self.id, 'logs')

    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def _get_weight_initializer(self):
        return tf.contrib.layers.xavier_initializer()

    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def _get_bias_initializer(self):
        return tf.contrib.layers.xavier_initializer() if self.use_bias else tf.zeros_initializer()

    def _init_summaries(self):
        tf.summary.histogram('input', self._states, collections=[self.name])
        tf.summary.histogram('rewards', self._activation_out, collections=[self.name])
        tf.summary.histogram('y', self._y, collections=[self.name])

    def _prepare_log_dir(self):
        if tf.gfile.Exists(self.log_dir_path):
            tf.gfile.DeleteRecursively(self.log_dir_path)
        tf.gfile.MakeDirs(self.log_dir_path)

    def run(self, tensors, states=None, probabilities=None):
        feed_dict = self._create_feed_dict(states, probabilities=probabilities)

        return self.session.run(tensors, feed_dict=feed_dict)

    @property
    def gradient_of_input_wrt_activation(self):
        grad_tensor = tf.gradients(self._activation_out, self._states)
        return grad_tensor[0]

    def predict(self, states):
        feed_dict = self._create_feed_dict(states)

        return self.session.run(self._activation_out, feed_dict=feed_dict)

    def update(self, states, probabilities, feed_dict_processor=identity):
        if self.visualize:
            TensorVisualizer.instance.cache_inputs(states)
            TensorVisualizer.instance.cache_input_gradients(self, states)
            TensorVisualizer.instance.cache_gradients(self, states, probabilities)

        self._update(states, probabilities, feed_dict_processor=feed_dict_processor)

        if self.visualize:
            TensorVisualizer.instance.cache_kernels(self)
            TensorVisualizer.instance.cache_activations(self, states)

            TensorVisualizer.instance.pack_cache()

    def _update(self, states, probabilities, feed_dict_processor=identity):
        feed_dict = self._create_feed_dict(states, probabilities=probabilities)
        feed_dict = feed_dict_processor(feed_dict)

        params = [self._loss_tensor, self._optimizer_tensor, self._global_step]
        if hasattr(self, 'merged_summery') and self.merged_summery is not None:
            params += [self.merged_summery]
            loss, _, step, summary = self.session.run(params, feed_dict=feed_dict)
            self.train_writer.add_summary(summary, global_step=step)
        else:
            loss, _, _ = self.session.run(params, feed_dict=feed_dict)
        return loss

    def _create_feed_dict(self, states, probabilities=None):
        are_invalid_inputs = states is None and probabilities is not None
        assert not are_invalid_inputs

        if states is not None and len(states) > 0:
            states = self._process_input(states)
            feed_dict = {self._states: states}

            if probabilities is not None:
                feed_dict[self._y] = probabilities
        else:
            feed_dict = None

        return feed_dict
