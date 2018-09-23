import numpy as np
import tensorflow as tf
from collections import deque

import ene
from models.dqn import DQN
from utils.functions import noop
from utils.logger import Logger
from learn.utils.progress_utils import ClearManager
from learn.utils.environment_player import simulate_play

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5


def train(env, episodes=50000, action_callback=noop, ene_mode='e-greedy'):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        input_dim = np.prod(env.state_shape)
        output_dim = env.action_size

        # target_dqn is slightly behind main_dqn(therefore, target_dqn has slightly old parameters),
        # so that training is done on stationary target.
        main_dqn = DQN(sess, input_dim, output_dim, hidden_sizes=[32, 16],
                       learning_rate=1e-3, name='main', log_name_postfix='s' if env.is_stochastic else 'd')
        target_dqn = DQN(sess, input_dim, output_dim, hidden_sizes=[32, 16],
                         learning_rate=1e-3, name='target', write_tensor_log=False)
        tf.global_variables_initializer().run()

        with main_dqn, target_dqn:
            _train(sess, main_dqn, target_dqn, env, episodes, action_callback, ene_mode)


def _train(sess, main_dqn, target_dqn, env, episodes, action_callback, ene_mode):
    logger = Logger(main_dqn.log_dir_name)
    clear_manager = ClearManager()

    select = ene.modes[ene_mode]
    replay_memory = deque(maxlen=50000)

    copy_operations = target_dqn.build_copy_variables_from_operation(main_dqn)
    sess.run(copy_operations)

    action_spec = {
        'count': env.action_size,
        'generator': lambda s: main_dqn.predict(s)[0],
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        clear_manager.do_soft_reset()

        Q = main_dqn.predict(range(main_dqn.input_dim))
        while not done:
            action = select(episode, state, action_spec)
            actual_action, new_state, reward, done = env.step(action)

            clear_manager.save_reward(reward)
            replay_memory.append((state, action, reward, new_state, done))

            if len(replay_memory) > BATCH_SIZE:
                DQN.replay_train(main_dqn, target_dqn, replay_memory, DISCOUNT_RATE, minibatch_size=BATCH_SIZE)
            if env.steps % TARGET_UPDATE_FREQUENCY == 0:
                sess.run(copy_operations)

            state = new_state

            action_callback(env, Q, episode, state, action, actual_action)

        summary = env.get_summary_lines(Q)
        logger.log_summary(episode, summary)

        clear_manager.update_last_100_games_rewards()
        clear_manager.print_progress(episode, env.steps)
        if clear_manager.has_cleared(env):
            clear_manager.print_cleared_message(episode)
            simulate_play(env, main_dqn)
            break
