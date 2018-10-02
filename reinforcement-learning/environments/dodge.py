from bidict import bidict

import numpy as np

from games.dodge import Dodge, SCREEN_RECT, get_grayscale_frame, ensure_draw
from environments.environment import Environment
from utils.functions import *


def disable_pygame_display():
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"


class DodgeEnv(Dodge, Environment):
    screen_size = (SCREEN_RECT.width, SCREEN_RECT.height)
    actions = bidict({
        0: (0, 0),
        1: (0, -1),
        2: (1, -1),
        3: (1, 0),
        4: (1, 1),
        5: (0, 1),
        6: (-1, 1),
        7: (-1, 0),
        8: (-1, -1),
    })

    # TODO: Adjust threshold
    def __init__(self, threshold=500, headless_mode=False, is_stochastic=True):
        disable_pygame_display() if headless_mode else noop
        Dodge.__init__(self, move_zombies_randomly=is_stochastic)
        Environment.__init__(self, list(DodgeEnv.screen_size) + [1], len(DodgeEnv.actions), threshold,
                             is_stochastic=is_stochastic, network_mode=Environment.convolution)

    def step(self, action) -> tuple:
        unflattened_action = self.unflatten_action(action)
        collided_zombies = self.tick(*unflattened_action)

        state = self.frame
        done = len(collided_zombies) > 0
        reward = 1 if not done else -100
        self.steps += 1

        if self.steps > self.threshold * 2:
            done = True
            reward = 1

        return action, state, reward, done

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.reset_game()

        self.state = self.frame
        self.steps = 0

        return self.state

    def flatten_state(self, unflattened_state):
        raise NotImplementedError

    def unflatten_state(self, flattened_state):
        raise NotImplementedError

    def flatten_action(self, unflattened_action):
        return DodgeEnv.actions.inv[unflattened_action]

    def unflatten_action(self, flattened_action):
        return DodgeEnv.actions[flattened_action]

    @property
    def frame(self):
        ensure_draw()
        return np.expand_dims(get_grayscale_frame(), axis=-1)
