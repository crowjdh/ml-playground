from abc import ABC, abstractmethod


class Environment(ABC):
    dense = 'dense'
    convolution = 'conv'

    def __init__(self, state_shape, action_size, threshold, is_stochastic=True, network_mode=dense):
        self.is_stochastic = is_stochastic
        self.threshold = threshold
        self.reward_processor = None
        self.state_shape = state_shape
        self.action_size = action_size
        self.network_mode = network_mode

    @property
    def id(self):
        return '{}_{}'.format(self.__class__.__name__, 's' if self.is_stochastic else 'd')

    @abstractmethod
    def step(self, action) -> tuple:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def flatten_state(self, unflattened_state):
        pass

    @abstractmethod
    def unflatten_state(self, flattened_state):
        pass

    @abstractmethod
    def flatten_action(self, unflattened_action):
        pass

    @abstractmethod
    def unflatten_action(self, flattened_action):
        pass
