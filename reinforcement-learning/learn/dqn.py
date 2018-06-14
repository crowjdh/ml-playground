import numpy as np
import tensorflow as tf
from collections import deque

import ene
from models.dqn import DQN
from utils.functions import noop
from utils.logger import Logger

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5


def simulate_play(env, dqn, count=10):
    for i in range(count):
        state = env.reset()
        done = False
        reward_sum = 0

        while not done:
            action = np.argmax(dqn.predict(state))
            actual_action, new_state, reward, done = env.step(action)
            reward_sum += reward
            state = new_state

        print("reward_sum: {}".format(reward_sum))


def log_Q(Q, episode, env, logger):
    summary = env.get_summary_lines(Q)
    logger.log('Episode {}\n'.format(episode))
    logger.log(summary)


def train(env, episodes=5000, action_callback=noop, ene_mode='e-greedy'):
    select = ene.modes[ene_mode]
    replay_memory = deque(maxlen=50000)
    last_100_games_rewards = deque(maxlen=100)

    input_dim = np.prod(env.state_shape)
    output_dim = env.action_size

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        # target_dqn is slightly behind main_dqn(therefore, target_dqn has slightly old parameters),
        # so that training is done on stationary target.
        main_dqn = DQN(sess, input_dim, output_dim, hidden_sizes=[32, 16], learning_rate=1e-3, name='main')
        target_dqn = DQN(sess, input_dim, output_dim, hidden_sizes=[32, 16], learning_rate=1e-3, name='target',
                         write_tensor_log=False)
        logger = Logger(main_dqn.log_name)

        with main_dqn, target_dqn:
            tf.global_variables_initializer().run()

            copy_operations = target_dqn.build_copy_variables_from_operation(main_dqn)
            sess.run(copy_operations)

            action_spec = {
                'count': env.action_size,
                'generator': lambda s: main_dqn.predict(s)[0],
            }

            for episode in range(episodes):
                state = env.reset()
                done = False
                reward_sum = 0
                steps = 0

                Q = main_dqn.predict(range(16))
                log_Q(Q, episode, env, logger)
                while not done:
                    action = select(episode, state, action_spec)
                    actual_action, new_state, reward, done = env.step(action)

                    if done and reward != 1:
                        reward = -1
                    reward_sum += reward

                    replay_memory.append((state, action, reward, new_state, done))

                    if len(replay_memory) > BATCH_SIZE:
                        DQN.replay_train(main_dqn, target_dqn, replay_memory, DISCOUNT_RATE, minibatch_size=BATCH_SIZE)
                    if steps % TARGET_UPDATE_FREQUENCY == 0:
                        sess.run(copy_operations)

                    state = new_state
                    steps += 1

                    action_callback(env, Q, episode, state, action, actual_action)

                avg_reward = np.mean(last_100_games_rewards)
                avg_reward = (avg_reward + 1) / 2
                print("Episode: {:5.0f}, steps: {:5.0f}, rewards: {:2.0f}, avg_reward:{:6.2f}"
                      .format(episode, steps, reward_sum, avg_reward))
                last_100_games_rewards.append(reward_sum)

                if len(last_100_games_rewards) == last_100_games_rewards.maxlen:
                    if avg_reward > env.threshold:
                        print(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                        simulate_play(env, main_dqn)
                        break
