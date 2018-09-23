from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, threshold, is_stochastic=True):
        self.is_stochastic = is_stochastic
        self.threshold = threshold
        self.reward_processor = None

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
