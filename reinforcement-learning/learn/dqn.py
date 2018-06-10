import numpy as np
import tensorflow as tf
from collections import deque

import ene
from models.dqn import DQN
from environments.frozen_lake import FrozenLake
from utils.functions import noop


def train(lake, episode_size=10000, action_callback=noop, ene_mode='e-greedy'):
    select = ene.modes[ene_mode]
    replay_memory = deque(maxlen=50000)

    # input_dim = np.prod(lake.lake_size)
    input_dim = 1
    output_size = lake.action_size
    discount_factor = 0.9

    # These constant values may cause performance issue.
    # Move replay train outside while loop when increasing greater than 1.
    replay_learn_freq = 10
    update_target_dqn_freq = replay_learn_freq * 5
    learn_start = 10

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        # target_dqn is slightly behind main_dqn(therefore, target_dqn has slightly old parameters),
        # so that training is done on stationary target.
        main_dqn = DQN(sess, input_dim, output_size, hidden_sizes=[20, 10], learning_rate=1e-1, name='main', Q_formatter=lake.Q_formatter)
        target_dqn = DQN(sess, input_dim, output_size, hidden_sizes=[20, 10], learning_rate=1e-1, name='target', Q_formatter=lake.Q_formatter, write_tensor_log=False)
        with main_dqn, target_dqn:
            tf.global_variables_initializer().run()

            copy_operations = target_dqn.build_copy_variables_from_operation(main_dqn)
            sess.run(copy_operations)

            # if True:
            #     states = [
            #         # 0,
            #         # 0,
            #         14]
            #     rewards = [
            #         # [-1, 0, 0, 0],
            #         # [0, 0, 0, 0],
            #         [0, 100, 0, 0]
            #     ]
            #     loss = main_dqn.update(states, rewards)
            #     Q_after = main_dqn.get_Q()
            #     return

            reward_sum = None
            best_reward_sum = None
            for episode_idx in range(episode_size):
                state = FrozenLake.flatten_state(lake.reset())
                done = False

                Q = main_dqn.get_Q().reshape(list(lake.lake_size) + [lake.action_size])
                while not done:
                    action = select(episode_idx, main_dqn.predict(state)[0])
                    actual_action, new_state, reward, done = lake.step(action)
                    if done:
                        if reward != 1:
                            reward = -10
                        else:
                            reward = 10

                    new_state = FrozenLake.flatten_state(new_state)
                    replay_memory.append((state, action, reward, new_state, done))
                    if not reward_sum:
                        reward_sum = reward
                    else:
                        reward_sum += reward

                    state = new_state
                    # steps += 1

                    unflattened_state = FrozenLake.unflatten_state(state)
                    action_callback(lake, Q, episode_idx, unflattened_state, action, actual_action)

                if not best_reward_sum:
                    best_reward_sum = reward_sum

                if episode_idx >= learn_start and episode_idx % replay_learn_freq == 0:
                    main_dqn.log('Episode {}\n'.format(episode_idx))
                    main_dqn.log('Main DQN(before):')
                    Q_before = main_dqn.get_Q()
                    main_dqn.log(main_dqn.Q_formatter(Q_before))

                    loss = DQN.replay_train(main_dqn, target_dqn, replay_memory, discount_factor, minibatch_size=500)

                    main_dqn.log('Main DQN(after):')
                    Q_after = main_dqn.get_Q()
                    main_dqn.log(main_dqn.Q_formatter(Q_after))

                    main_dqn.log('Main DQN diff:')
                    main_dqn.log(main_dqn.Q_formatter(Q_after - Q_before))

                    main_dqn.log('='*80 + '\n')

                    # if reward_sum > best_reward_sum:
                    #     best_reward_sum = reward_sum
                    #     sess.run(copy_operations)

                    print("episode {:4d}\treward_sum: {}, loss: {}".format(episode_idx, reward_sum, loss))
                    reward_sum = 0
                    # sess.run(copy_operations)

                if episode_idx >= learn_start and episode_idx % update_target_dqn_freq == 0:
                    sess.run(copy_operations)
