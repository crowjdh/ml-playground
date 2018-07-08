import numpy as np
from collections import deque


class ClearManager:
    def __init__(self, print_progress_frequency=1):
        self.last_100_games_rewards = deque(maxlen=100)
        self.print_progress_frequency = print_progress_frequency
        self.do_soft_reset()

    def save_reward(self, reward):
        self.rewards.append(reward)
        self.reward_sum += reward

    def update_last_100_games_rewards(self):
        self.last_100_games_rewards.append(self.reward_sum)

    # noinspection PyAttributeOutsideInit
    def do_soft_reset(self):
        self.rewards = []
        self.reward_sum = 0

    @property
    def average_reward(self):
        if len(self.last_100_games_rewards) < self.last_100_games_rewards.maxlen:
            return 0

        return np.mean(self.last_100_games_rewards)

    def has_cleared(self, env):
        return self.average_reward > env.threshold

    def print_progress(self, episode, steps):
        if episode % self.print_progress_frequency == 0:
            print("Episode: {:5.0f}, steps: {:5.0f}, rewards: {:2.0f}, avg_reward:{:6.2f}"
                  .format(episode, steps, self.reward_sum, self.average_reward))

    def print_cleared_message(self, episode):
        print(f"Game Cleared in {episode} episodes with avg reward {self.average_reward}")
