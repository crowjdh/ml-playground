import os
from abc import ABC, abstractmethod
from enum import Enum
import pathlib

import numpy as np

from utils import numpy_writer
from utils.singleton import Singleton

VISUALIZER_ROOT_DIR_PATH = '.tensor_visualizer'
LAYER_INFO_SEPARATOR = ','


def _create_id(network, infix, idx):
    return '{}-{}{}'.format(network.id, infix, idx)


class TensorVisualizer(metaclass=Singleton):
    def __init__(self):
        self._id = "NA"
        self.root_tensor_node = None
        self.current_tensor_node = None
        self.history = []
        self.save_for_every_n_updates = 300
        self.update_item_count = 5
        self._update_count = 0
        self._tensor_history_cache = None
        self._inputs_cache = None
        self._input_gradients_cache = None

    def setup(self):
        self._restore_update_count()
        self._save_layer_info()

    def _restore_update_count(self):
        snapshot_numbers = self.snapshot_numbers
        if not os.path.isdir(self.visualizer_dir_path) or len(snapshot_numbers) == 0:
            return
        last_snapshot_number = max(*self.snapshot_numbers)
        self._update_count = self.save_for_every_n_updates * last_snapshot_number

    def _save_layer_info(self):
        pathlib.Path(self.visualizer_dir_path).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(self.layer_info_file_path):
            return

        layer_names = [layer_type.value for layer_type in self.root_tensor_node.layer_types]
        with open(self.layer_info_file_path, 'w') as layer_info_file:
            layer_info_file.write(LAYER_INFO_SEPARATOR.join(layer_names))

    @property
    def snapshot_numbers(self):
        snapshot_numbers = []
        for file_path in os.listdir(self.visualizer_dir_path):
            try:
                snapshot_number_idx = file_path.rindex(self.visualizer_file_name) + len(self.visualizer_file_name)
                snapshot_number = int(file_path[snapshot_number_idx+1:])
                snapshot_numbers.append(snapshot_number)
            except ValueError:
                continue

        snapshot_numbers.sort()
        return snapshot_numbers

    @property
    def layer_info(self):
        if not os.path.exists(self.layer_info_file_path):
            return None

        with open(self.layer_info_file_path, 'r') as layer_info_file:
            layer_info_contents = layer_info_file.read()
            layer_names = layer_info_contents.split(LAYER_INFO_SEPARATOR)
            return [TensorNode.Layer(layer_name) for layer_name in layer_names]

    def add_dense_layer(self, layer, activation):
        self._add_layer(layer, activation, TensorNode.Layer.DENSE)

    def add_conv_layer(self, layer, activation):
        self._add_layer(layer, activation, TensorNode.Layer.CONV)

    def _add_layer(self, layer, activation, layer_type):
        new_node = TensorNode(layer, activation, layer_type)
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

    def cache_inputs(self, inputs):
        if not self._should_save_history():
            return

        # TODO: Should I cache preprocessed inputs?
        inputs = inputs[:self.update_item_count]
        self._inputs_cache = inputs[:self.update_item_count]

    def cache_input_gradients(self, network, inputs):
        if not self._should_save_history():
            return

        inputs = inputs[:self.update_item_count]
        grad_tensor = network.gradient_of_input_wrt_activation

        self._input_gradients_cache = network.run(grad_tensor, states=inputs)

    def cache_gradients(self, network, inputs, y):
        if not self._should_save_history():
            return

        inputs = inputs[:self.update_item_count]
        y = y[:self.update_item_count]

        layer_cnt = self.root_tensor_node.length
        input_cnt = len(inputs)
        all_gradients = [[] for _ in range(layer_cnt)]

        for idx in range(input_cnt):
            lhs = inputs[idx:idx + 1]
            rhs = y[idx:idx + 1]
            gradients = network.run(self.root_tensor_node.gradients, states=lhs, probabilities=rhs)
            for layer_idx in range(len(gradients)):
                all_gradients[layer_idx].append(gradients[layer_idx])
        all_gradients = [np.asarray(gradients) for gradients in all_gradients]

        self._cache_value(all_gradients)

    def cache_kernels(self, network):
        if not self._should_save_history():
            return

        kernels = network.run(self.root_tensor_node.kernels)
        self._cache_value(kernels)

    def cache_activations(self, network, inputs):
        if not self._should_save_history():
            return

        inputs = inputs[:self.update_item_count]

        conv_activations = network.run(self.root_tensor_node.activations, states=inputs)
        self._cache_value(conv_activations)

    def save_history(self):
        self._update_count += 1
        if not self._should_save_history():
            return

        visualizer_file_name = '{}_{}'.format(self.visualizer_file_name, self._history_index)

        numpy_writer.append_arrays(self.history, self.visualizer_dir_path, visualizer_file_name)
        self.history = []

    def load_histories(self, snapshot_indices):
        snapshot_numbers = np.asarray(self.snapshot_numbers)
        snapshot_indices = np.asarray(snapshot_indices)

        snapshot_numbers = snapshot_numbers[snapshot_indices]

        return [self.load_history(snapshot_number)[0] for snapshot_number in snapshot_numbers]

    def load_history(self, snapshot_idx):
        visualizer_file_name = '{}_{}'.format(self.visualizer_file_name, snapshot_idx)

        self.history = numpy_writer.load_arrays(self.visualizer_dir_path, visualizer_file_name)
        return self.history

    def _cache_value(self, value):
        if self._tensor_history_cache is None:
            self._tensor_history_cache = []
        self._tensor_history_cache.append(value)

    def pack_cache(self):
        if not self._should_save_history():
            return

        layers = self.collect_layer_values()
        self.history.append([self._inputs_cache, self._input_gradients_cache, layers])

        self._tensor_history_cache = None
        self._inputs_cache = None
        self._input_gradients_cache = None

    def collect_layer_values(self):
        layers = []
        for layer_idx in range(self.root_tensor_node.length):
            layer = []
            for type_idx in range(len(self._tensor_history_cache)):
                layer.append(self._tensor_history_cache[type_idx][layer_idx])
            layers.append(layer)

        return layers

    def _should_save_history(self):
        return self._update_count % self.save_for_every_n_updates == 0

    @property
    def _history_index(self):
        return int(self._update_count / self.save_for_every_n_updates) if self._should_save_history() else -1

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

    @property
    def layer_info_file_path(self):
        return os.path.join(self.visualizer_dir_path, 'layer_info')


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

    @property
    def length(self):
        _length = 0

        def collector(_):
            nonlocal _length
            _length += 1

        self.traverse(collector)

        return _length

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
    class Layer(Enum):
        DENSE = 'dense'
        CONV = 'conv'

        @classmethod
        def values(cls):
            return [layer.value for layer in cls]

    def __init__(self, layer, activation, layer_type):
        super(TensorNode, self).__init__()

        self.layer = layer
        self.activation = activation
        self.gradient = None
        self.layer_type = layer_type

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

    @property
    def layer_types(self):
        return self.collect(lambda node: node.layer_type)
