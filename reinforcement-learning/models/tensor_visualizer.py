import os
from abc import ABC, abstractmethod

from utils import numpy_writer
from utils.singleton import Singleton

VISUALIZER_ROOT_DIR_PATH = '.tensor_visualizer'


def _create_id(network, infix, idx):
    return '{}-{}{}'.format(network.id, infix, idx)


class TensorVisualizer(metaclass=Singleton):
    def __init__(self):
        self._id = "NA"
        self.root_tensor_node = None
        self.current_tensor_node = None
        self.history = []

    def add_layer(self, layer, activation):
        new_node = TensorNode(layer, activation)
        if self.root_tensor_node is None:
            self.root_tensor_node = new_node
        else:
            self.current_tensor_node.add_next(new_node)
        self.current_tensor_node = new_node

    def collect_gradients(self, grads_and_vars):
        vars_and_grads = {var: grad for grad, var in grads_and_vars}

        def action(node):
            node.gradient = vars_and_grads[node.layer.kernel]
        self.root_tensor_node.traverse(action)

    def expand_history(self):
        self.history.append([])

    def cache_gradients(self, network, inputs, y):
        gradients = network.run(self.root_tensor_node.gradients, states=inputs, probabilities=y)
        self._cache_value(gradients)

    def cache_kernels(self, network):
        kernels = network.run(self.root_tensor_node.kernels)
        self._cache_value(kernels)

    def cache_activations(self, network, inputs):
        conv_activations = network.run(self.root_tensor_node.activations, states=inputs)
        self._cache_value(conv_activations)

    def save_history(self):
        numpy_writer.append_data(self.history, self.visualizer_dir_path, self.visualizer_file_name)
        # TODO: Enable below after validation
        # self.history = []

    def load_history(self):
        # TODO: Toggle after validation
        # self.history = numpy_writer.load_data(self.visualizer_dir_path, self.visualizer_file_name)
        return numpy_writer.load_data(self.visualizer_dir_path, self.visualizer_file_name)

    def _cache_value(self, value):
        self.history[-1].append(value)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def visualizer_dir_path(self):
        return os.path.join(VISUALIZER_ROOT_DIR_PATH, self.id)

    @property
    def visualizer_file_name(self):
        return 'visualizer'

    # TODO: Remove after validation
    # def pack_into_history(self):
    #     self.history.append(self.root_tensor_node.all_values)

    # def plot_conv_weights(self, network):
    #     kernels = network.run(self.root_tensor_node.kernels)
    #     for i, kernel in enumerate(kernels):
    #         conviz.plot_conv_weights(kernel, _create_id(network, i, 'kernel'))
    #
    # def plot_conv_output(self, network, inputs):
    #     conv_activations = network.run(self.root_tensor_node.activations, inputs)
    #     for i, conv_activation in enumerate(conv_activations):
    #         conviz.plot_conv_output(conv_activation, _create_id(network, i, 'activation'))
    #
    # def plot_gradients(self, network, states, y):
    #     for gradient_tensor in self.root_tensor_node.gradients:
    #         gradient = network.run(gradient_tensor, states, y)
    #         print(gradient)


class Node(ABC):
    def __init__(self):
        self.prev = None
        self.next = None

    def add_next(self, node):
        self.next = node
        node.prev = self

    @property
    @abstractmethod
    def values(self):
        pass

    @property
    def is_root(self):
        return self.prev is None

    @property
    def is_leaf(self):
        return self.next is None

    def traverse(self, action):
        current = self
        while current is not None:
            action(current)
            current = current.next

    def collect(self, selector):
        current = self
        collection = []
        while current is not None:
            collection.append(selector(current))
            current = current.next

        return collection


class TensorNode(Node):
    def __init__(self, layer, activation):
        super(TensorNode, self).__init__()

        self.layer = layer
        self.activation = activation
        self.gradient = None

    @property
    def values(self):
        return [self.layer, self.activation, self.gradient]

    @property
    def layers(self):
        return self.collect(lambda node: node.layer)

    @property
    def kernels(self):
        return self.collect(lambda node: node.layer.kernel)

    @property
    def activations(self):
        return self.collect(lambda node: node.activation)

    @property
    def gradients(self):
        return self.collect(lambda node: node.gradient)
