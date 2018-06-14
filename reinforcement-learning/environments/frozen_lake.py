import numpy as np
from collections import Iterable


# noinspection PyAttributeOutsideInit
class FrozenLake(object):
    def __init__(self, is_slippery=True, threshold=.8):
        self.pitfalls = [(e[0], e[1]) for e in np.array([[1, 1, 2, 3], [1, 3, 3, 0]]).T]
        self.goal = 3, 3

        self.state_shape = (4, 4)
        self.lake = np.zeros(self.state_shape, dtype=np.int8)
        self.lake[self.goal] = 1
        self.is_slippery = is_slippery
        self.threshold = threshold

        self.start = 0, 0
        self.reset()
        # 0123 -> urdl
        self.action_position_map = [
            lambda: (self.state[0] - 1, self.state[1]),
            lambda: (self.state[0], self.state[1] + 1),
            lambda: (self.state[0] + 1, self.state[1]),
            lambda: (self.state[0], self.state[1] - 1)
        ]
        self.action_size = len(self.action_position_map)
        self.reward_processor = None

    def step(self, action):
        if self.is_slippery:
            # When lake is slippery, you might end up moving into
            # one of 3 possible states(with uniform distribution), which is:
            # 1. Desired action
            # 2. counter clockwise to 1
            # 3. clockwise to 1
            #
            # For example, if your bot choosed to move right, you can end up moving toward right(which is desirable),
            # or toward up/down if your bot slipped.
            #
            #  ---------------
            #         Up   <-|    2
            #                |
            #   Left       Right  1
            #                |
            #        Down  <-|    3
            #
            # This is how FrozenLake in OpenAI works.
            # See (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py#L101)

            action_candidates = np.array([action - 1, action, action + 1]) % 4
            action = np.random.choice(action_candidates)

        state = self.action_position_map[action]()
        self.state = self.clamp(state)

        done = self.state == self.goal or self.state in self.pitfalls
        reward = self.lake[self.state]

        flattened_state = self._get_flattened_state()
        if self.reward_processor:
            reward = self.reward_processor(action, flattened_state, reward, done)

        return action, flattened_state, reward, done

    def reset(self):
        self.state = self.start

        return self._get_flattened_state()

    def random_reset(self):
        state = np.random.choice(np.prod(self.state_shape))
        self.state = self.unflatten_state(state)

        return self.state

    def clamp(self, position):
        new_y = max(min(position[0], self.state_shape[0] - 1), 0)
        new_x = max(min(position[1], self.state_shape[1] - 1), 0)

        return new_y, new_x

    def _get_flattened_state(self):
        return self.state[0] * 4 + self.state[1]

    def unflatten_state(self, flattened_state):
        return flattened_state // self.state_shape[0], flattened_state % self.state_shape[1]

    def get_summary_lines(self, Q):
        from functools import reduce

        lines = ['' for _ in range(16)]
        for i in range(len(lines)):
            level = i // 4
            row_in_level = i % 4

            if row_in_level == 3:
                continue

            Q_start_idx = level * 4
            Q_indices = [Q_start_idx + j for j in range(4)]
            items = None
            out_format = None
            if row_in_level == 0:
                out_format = '{:^20.2f}'
                items = Q[Q_indices, 0]
            elif row_in_level == 1:
                out_format = '{:^8.2f}{:^4s}{:^8.2f}'
                items = []
                for Q_idx in Q_indices:
                    unflattened_state = self.unflatten_state(Q_idx)
                    state_msg = ''
                    if unflattened_state in self.pitfalls:
                        state_msg = 'XX'
                    elif unflattened_state == self.goal:
                        state_msg = '!!'
                    right, left = Q[Q_idx, [False, True, False, True]]
                    items.append([left, state_msg, right])
            elif row_in_level == 2:
                out_format = '{:^20.2f}'
                items = Q[Q_indices, 2]

            lines[i] = reduce(lambda lhs, rhs: lhs + rhs,
                              map(lambda e: out_format.format(*e) if isinstance(e, Iterable) else out_format.format(e), items))

        return lines
