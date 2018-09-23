import numpy as np
import tensorflow as tf

from utils.functions import noop
from utils.logger import Logger
from learn.utils.progress_utils import ClearManager
from learn.utils.environment_player import simulate_play

# Toggle comment on 2 lines below to use alternative strategy
from models.policy_gradient_net import PGN
# from models.policy_gradient_net_alternative import PGNAlternative as PGN

policy_gradient_update_frequency = 30
epsilon = 1e-10  # For numeric stability


def train(env, episodes=50000, action_callback=noop):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        input_dim = np.prod(env.state_shape)
        output_dim = env.action_size
        # Using bias on pg seems to make model to converge slower.
        network = PGN(sess, input_dim, output_dim, (128,),
                      learning_rate=1e-2, use_bias=False, name='pgn', log_name_postfix='s' if env.is_stochastic else 'd')
        tf.global_variables_initializer().run()

        with network:
            _train(network, env, episodes, action_callback)
        pass
    pass


def _train(network, env, episodes, action_callback):
    logger = Logger(network.log_dir_name)
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

        if episode % policy_gradient_update_frequency == policy_gradient_update_frequency - 1:
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
    # probe                      : 0.6
    #
    # =>
    # probe                           :               0.6
    # cumulative_probabilities        : [0.3,   0.5,       0.9,  1.0]
    # cumulative_probabilities > probe: [False, False,     True, True]
    #
    # (cumulative_probabilities > probe).argmax(): 2(index of first occurring True)
    cumulative_probabilities = np.cumsum(probabilities)
    probe = np.random.rand()
    return (cumulative_probabilities > probe).argmax()


class History:
    def __init__(self, discount_rate=0.99):
        self.reset()
        self.state = []
        self.action = []
        self.probabilities = []
        self.discounted_reward = []
        self.rewards = []
        self.reward_sum = 0
        self.discount_rate = discount_rate

    def save(self, state, action, probabilities, reward):
        self.state.append(state)
        self.action.append(action)
        self.probabilities.append(probabilities)
        self.rewards.append(reward)
        self.reward_sum += reward

    def update_discounted_rewards(self):
        # self.rewards                      : [     0,      0,       0,        0,        0,       -1]
        #
        # Unnormalized discounted_rewards(α): [-0.951, -0.961,  -0.970,   -0.980,   -0.990,   -1.000]
        # α - mean(α)                       : [ 0.024,  0.015,   0.005,   -0.005,   -0.015,   -0.025]
        # (α - mean(α)) / std(α)            : [ 1.454,  0.880,   0.301,   -0.285,   -0.876,   -1.474]
        discounted_rewards = self._calculate_discounted_rewards()
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + epsilon)
        self.discounted_reward.append(discounted_rewards)

        self.rewards = []

    def pack_discounted_reward_history(self):
        self.discounted_reward = np.concatenate(self.discounted_reward)[:, np.newaxis]

    def _calculate_discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float)
        prev_sum = 0
        for i in reversed(range(len(self.rewards))):
            prev_sum = self.discount_rate * prev_sum + self.rewards[i]
            discounted_rewards[i] = prev_sum

        return discounted_rewards

    def reset(self):
        self.state = []
        self.action = []
        self.probabilities = []
        self.discounted_reward = []
        self.rewards = []
        self.reward_sum = 0
