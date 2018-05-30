import numpy as np
from utils import rand_argmax
from env import Lake

def train(stdscr, pre_action_callback):
    lake = Lake()
    action_size = len(lake.action_position_map)
    Q_sizes = list(lake.lake_size) + [action_size]
    Q = np.zeros(np.prod(Q_sizes), dtype=np.int8).reshape(Q_sizes)
    rewards = []

    episode_size = 100
    for episode in range(episode_size):
        state = lake.reset()
        action = None
        done = False
        total_reward = 0

        while not done:
            pre_action_callback(stdscr, lake.lake, Q, episode, state, action)

            possible_actions = Q[state]
            action = rand_argmax(possible_actions)
            new_state, reward, done = lake.step(action)
            Q[state][action] = reward + np.max(Q[new_state])
            state = new_state
            total_reward += reward

        rewards.append(total_reward)

    return rewards
