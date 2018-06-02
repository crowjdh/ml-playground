import numpy as np

class Lake(object):
    def __init__(self, is_slippery=True):
        pitfalls = [1, 1, 2, 3],\
                   [1, 3, 3, 0]
        self.pitfalls = [(e[0], e[1]) for e in np.array(pitfalls).T]
        self.goal = 3, 3

        self.lake_size = (4, 4)
        self.lake = np.zeros(self.lake_size, dtype=np.int8)# height * width
        self.lake[self.goal] = 1
        self.is_slippery = is_slippery

        self.start = 0, 0
        self.reset()
        # 0123 -> urdl
        self.action_position_map = [
            lambda: self.clamp((self.state[0] - 1, self.state[1])),
            lambda: self.clamp((self.state[0], self.state[1] + 1)),
            lambda: self.clamp((self.state[0] + 1, self.state[1])),
            lambda: self.clamp((self.state[0], self.state[1] - 1))
        ]

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
            # This is how FrozenLake in OpenAI works.(https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py#L101)

            action_candidates = np.array([action - 1, action, action + 1]) % 4
            action = np.random.choice(action_candidates)

        self.state = self.action_position_map[action]()

        reward = self.lake[self.state]
        done = self.state == self.goal or self.state in self.pitfalls

        return action, self.state, reward, done

    def reset(self):
        self.state = self.start

        return self.state

    def clamp(self, position):
        new_y = max(min(position[0], self.lake_size[0] - 1), 0)
        new_x = max(min(position[1], self.lake_size[1] - 1), 0)

        return (new_y, new_x)
