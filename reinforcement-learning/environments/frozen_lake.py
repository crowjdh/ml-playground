import numpy as np
from collections import Iterable

from environments.environment import Environment


class FrozenLake(Environment):
    size = (4, 4)
    start = 0, 0
    action_size = 4

    def __init__(self, is_stochastic=True, threshold=.8, penalty_on_going_out=False):
        super().__init__(FrozenLake.size, FrozenLake.action_size, threshold, is_stochastic=is_stochastic)

        self.pitfalls = [(e[0], e[1]) for e in np.array([[1, 1, 2, 3], [1, 3, 3, 0]]).T]
        self.goal = 3, 3

        self.lake = np.zeros(self.state_shape, dtype=np.int8)
        self.lake[self.goal] = 1
        self.penalty_on_going_out = penalty_on_going_out

        self.reset()
        # 0123 -> urdl
        self.action_position_map = [
            lambda: (self.state[0] - 1, self.state[1]),
            lambda: (self.state[0], self.state[1] + 1),
            lambda: (self.state[0] + 1, self.state[1]),
            lambda: (self.state[0], self.state[1] - 1)
        ]

        assert len(self.action_position_map) == FrozenLake.action_size

    def step(self, action):
        if self.is_stochastic:
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
        self.state = self._clamp(state)
        self.steps += 1

        if self.penalty_on_going_out:
            has_went_out = self.state != state
        else:
            has_went_out = False
        done = has_went_out or self.state == self.goal or self.state in self.pitfalls
        reward = -1 if has_went_out else self.lake[self.state]

        flattened_state = self.flattened_state
        if self.reward_processor:
            reward = self.reward_processor(action, flattened_state, reward, done)

        # TODO: Consider adding maximum step count to accommodate infinite loop

        return action, flattened_state, reward, done

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.state = FrozenLake.start
        self.steps = 0

        return self.flattened_state

    def _clamp(self, position):
        new_y = max(min(position[0], self.state_shape[0] - 1), 0)
        new_x = max(min(position[1], self.state_shape[1] - 1), 0)

        return new_y, new_x

    def flatten_state(self, unflattened_state):
        return unflattened_state[0] * 4 + unflattened_state[1]

    def unflatten_state(self, flattened_state):
        return flattened_state // self.state_shape[0], flattened_state % self.state_shape[1]

    def flatten_action(self, unflattened_action):
        return unflattened_action

    def unflatten_action(self, flattened_action):
        return flattened_action

    @property
    def flattened_state(self):
        return self.flatten_state(self.state)

    @property
    def possible_states(self):
        return list(range(FrozenLake.size[0] * FrozenLake.size[1]))

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
