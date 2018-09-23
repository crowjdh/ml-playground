from bidict import bidict

from games.dodge import Dodge, get_grayscale_frame
from environments.environment import Environment


class DodgeEnv(Dodge, Environment):
    actions = bidict({
        0: (0, -1),
        1: (1, -1),
        2: (1, 0),
        3: (1, 1),
        4: (0, 1),
        5: (-1, 1),
        6: (-1, 0),
        7: (-1, -1)
    })

    def __init__(self, threshold=.8):
        Dodge.__init__(self)
        Environment.__init__(self, threshold)

    def step(self, action) -> tuple:
        unflattened_action = self.unflatten_action(action)
        collided_zombies = self.tick(*unflattened_action)

        state = get_grayscale_frame()
        done = len(collided_zombies) > 0
        reward = 1 if not done else -100

        return action, state, reward, done

    def reset(self):
        self.reset_game()

        return get_grayscale_frame()

    def flatten_state(self, unflattened_state):
        raise NotImplementedError

    def unflatten_state(self, flattened_state):
        raise NotImplementedError

    def flatten_action(self, unflattened_action):
        return DodgeEnv.actions.inv[unflattened_action]

    def unflatten_action(self, flattened_action):
        return DodgeEnv.actions[flattened_action]
