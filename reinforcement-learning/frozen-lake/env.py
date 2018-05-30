import numpy as np

class Lake(object):
    def __init__(self):
        pitfalls = [1, 3, 1],\
                   [1, 2, 3]
        goal = 3, 3

        self.lake_size = (4, 4)
        self.lake = np.zeros(self.lake_size, dtype=np.int8)# height * width
        self.lake[pitfalls] = -1
        self.lake[goal] = 1

        self.start = 0, 0
        self.reset()
        # 0123 -> urdl
        self.action_position_map = [
            lambda: (self.state[0] - 1, self.state[1]),
            lambda: (self.state[0], self.state[1] + 1),
            lambda: (self.state[0] + 1, self.state[1]),
            lambda: (self.state[0], self.state[1] - 1)
        ]
        pass

    def step(self, action):
        new_pos = self.action_position_map[action]()
        
        new_y = max(min(new_pos[0], self.lake_size[0] - 1), 0)
        new_x = max(min(new_pos[1], self.lake_size[1] - 1), 0)
        went_out = new_y != new_pos[0] or new_x != new_pos[1]

        self.state = new_y, new_x
        reward = -1 if went_out else self.lake[self.state]
        done = reward != 0

        return self.state, reward, done

    def reset(self):
        self.state = self.start

        return self.state
