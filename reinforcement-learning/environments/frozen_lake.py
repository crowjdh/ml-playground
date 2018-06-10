import numpy as np


# noinspection PyAttributeOutsideInit
class FrozenLake(object):
    def __init__(self, is_slippery=True):
        self.pitfalls = [(e[0], e[1]) for e in np.array([[1, 1, 2, 3], [1, 3, 3, 0]]).T]
        self.goal = 3, 3

        self.lake_size = (4, 4)
        self.lake = np.zeros(self.lake_size, dtype=np.int8)# height * width
        self.lake[self.goal] = 1
        self.is_slippery = is_slippery

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

        # went_out
        state = self.action_position_map[action]()
        self.state = self.clamp(state)

        reward = self.lake[self.state]
        done = self.state == self.goal or self.state in self.pitfalls or state != self.state

        return action, self.state, reward, done

    def reset(self):
        self.state = self.start

        return self.state

    def clamp(self, position):
        new_y = max(min(position[0], self.lake_size[0] - 1), 0)
        new_x = max(min(position[1], self.lake_size[1] - 1), 0)

        return new_y, new_x

    @staticmethod
    def flatten_state(state):
        return state[0] * 4 + state[1]

    @staticmethod
    def unflatten_state(flattened_state):
        return flattened_state // 4, flattened_state % 4

    def Q_formatter(self, Q):
        from functools import reduce

        lines = ['' for _ in range(16)]
        for i in range(len(lines)):
            level = i // 4
            row_in_level = i % 4

            if row_in_level == 3:
                continue

            Q_start_idx = level * 4
            Q_indices = [Q_start_idx + j for j in range(4)]
            mask = [False] * 4
            items = None
            out_format = None
            if row_in_level == 0:
                out_format = '{:^20.2f}'
                items = Q[Q_indices, 0]
            elif row_in_level == 1:
                out_format = '{:^8.2f}{:^4s}{:^8.2f}'
                items = []
                for Q_idx in Q_indices:
                    unflattened_state = FrozenLake.unflatten_state(Q_idx)
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
                              map(lambda e: out_format.format(*e) if isinstance(e, list) else out_format.format(e), items))

        return lines
