import numpy as np
import tensorflow as tf

from models.policy_gradient_net import PGN
from utils.functions import noop
from utils.logger import Logger
from learn.utils.progress_utils import ClearManager
from learn.utils.environment_player import simulate_play

policy_gradient_update_frequency = 30


def train(env, episodes=50000, action_callback=noop):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        input_dim = np.prod(env.state_shape)
        output_dim = env.action_size
        # Using bias on pg seems to make model to converge slower.
        network = PGN(sess, input_dim, output_dim, (128,), learning_rate=1e-2, use_bias=False, name='pgn')
        tf.global_variables_initializer().run()

        with network:
            _train(network, env, episodes, action_callback)
        pass
    pass


def _train(network, env, episodes, action_callback):
    logger = Logger(network.log_name)
    history = History()
    clear_manager = ClearManager()

    for episode in range(episodes):
        state = env.reset()
        clear_manager.do_soft_reset()
        done = False

        Q = network.predict(range(network.input_dim))
        while not done:
            probabilities = network.predict(state)[0]
            action = _sample(probabilities)

            actual_action, new_state, reward, done = env.step(action)
            history.save(state, action, probabilities, reward)
            clear_manager.save_reward(reward)

            state = new_state

            action_callback(env, Q, episode, state, action, actual_action)

        history.update_discounted_rewards()

        if episode % policy_gradient_update_frequency == 29:
            history.pack_discounted_reward_history()
            network.perform_policy_gradient_update(history)
            history.reset()

            Q = network.predict(range(network.input_dim))
            summary = env.get_summary_lines(Q)
            logger.log_summary(episode, summary)

        clear_manager.update_last_100_games_rewards()
        clear_manager.print_progress(episode, env.steps)
        if clear_manager.has_cleared(env):
            clear_manager.print_cleared_message(episode)
            simulate_play(env, network)
            break


def _sample(probabilities):
    # Example:
    #
    # probabilities             : [0.3, 0.2, 0.4, 0.1]
    # cumulative_probabilities  : [0.3, 0.5, 0.9, 1.0]
    # coin                      : 0.6
    #
    # =>
    # coin:                                 0.6
    # cumulative_probabilities: [0.3, 0.5,       0.9, 1.0]
    #
    # cumulative_probabilities > coin: [False, False, True, True]
    # (cumulative_probabilities > coin).argmax(): 2(index of first occurring True)
    cumulative_probabilities = np.cumsum(probabilities)
    coin = np.random.rand()
    return (cumulative_probabilities > coin).argmax()


class History:
    def __init__(self, discount_rate=0.99):
        self.reset()
        self.state_history = []
        self.action_history = []
        self.probabilities_history = []
        self.discounted_reward_history = []
        self.rewards = []
        self.reward_sum = 0
        self.discount_rate = discount_rate

    def save(self, state, action, probabilities, reward):
        self.state_history.append(state)
        self.action_history.append(action)
        self.probabilities_history.append(probabilities)
        self.rewards.append(reward)
        self.reward_sum += reward

    def update_discounted_rewards(self):
        discounted_rewards = self._calculate_discounted_rewards()
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-10)
        self.discounted_reward_history.append(discounted_rewards)

        self.rewards = []

    def pack_discounted_reward_history(self):
        self.discounted_reward_history = np.concatenate(self.discounted_reward_history)[:, np.newaxis]

    def _calculate_discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float)
        prev_sum = 0
        for i in reversed(range(len(self.rewards))):
            prev_sum = self.discount_rate * prev_sum + self.rewards[i]
            discounted_rewards[i] = prev_sum

        return discounted_rewards

    def reset(self):
        self.state_history = []
        self.action_history = []
        self.probabilities_history = []
        self.discounted_reward_history = []
        self.rewards = []
        self.reward_sum = 0
