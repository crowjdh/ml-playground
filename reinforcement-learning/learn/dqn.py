from collections import deque

import numpy as np
import tensorflow as tf

import ene
from models.checkpoint import Checkpoint
from models.dqn_mixin import DQNMixin
from models.tensor_visualizer import TensorVisualizer
from utils.cleanup_util import CleanupHelper
from utils.functions import noop
from utils.logger import Logger
from learn.utils.progress_utils import ClearManager
from learn.utils.environment_player import simulate_play
from utils.replay_manager import ReplayManager
from utils.replay_memory_writer import ReplayMemoryWriter
from environments.environment import Environment

DISCOUNT_RATE = 0.99
FRAME_QUEUE_LENGTH = 1
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
CHECKPOINT_FREQUENCY = 10
RESIZE_CONV_FRAME_TO = [20, 20]


class FrameQueue:
    def __init__(self, queue_length):
        self.frames = deque(maxlen=queue_length)

    def fill(self, s):
        self.frames.append(s)

    def to_states(self):
        frames = list(self.frames)
        while len(frames) < self.frames.maxlen:
            frames.append(frames[-1])

        if np.isscalar(frames[0]):
            return np.asarray(frames)

        frames = np.asarray(frames)
        shape = list(frames.shape[1:-1]) + [frames.shape[0]*frames.shape[-1]]
        return frames.transpose([1, 2, 0, 3]).reshape(shape)

    @property
    def is_full(self):
        return len(self.frames) == self.frames.maxlen


def train(env, episodes=50000, action_callback=noop, ene_mode='e-greedy'):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        main_dqn, target_dqn = create_networks(sess, env)
        tf.global_variables_initializer().run()

        with main_dqn, target_dqn:
            _train(sess, main_dqn, target_dqn, env, episodes, action_callback, ene_mode)


def create_networks(sess, env):
    if env.network_mode == Environment.dense:
        return create_dense_networks(sess, env)
    elif env.network_mode == Environment.convolution:
        global FRAME_QUEUE_LENGTH
        FRAME_QUEUE_LENGTH = 4
        return create_conv_networks(sess, env)
    else:
        raise NotImplementedError("Network mode {} not implemented".format(env.network_mode))


def create_dense_networks(sess, env):
    from models.dense.dqn import DenseDQN as DQN

    input_dim = np.prod(env.state_shape)
    output_dim = env.action_size

    # target_dqn is slightly behind main_dqn(therefore, target_dqn has slightly old parameters),
    # so that training is done on stationary target.
    main_dqn = DQN(sess, env.id, input_dim, output_dim, hidden_sizes=[32, 16],
                   learning_rate=1e-3, name='main', visualize=False)
    target_dqn = DQN(sess, env.id, input_dim, output_dim, hidden_sizes=[32, 16],
                     learning_rate=1e-3, name='target', write_tensor_log=False, visualize=False)

    return main_dqn, target_dqn


def create_conv_networks(sess, env):
    from models.conv.dqn import ConvDQN as DQN

    input_shape = list(env.state_shape) + [FRAME_QUEUE_LENGTH]
    input_shape[:2] = env.resize_to = RESIZE_CONV_FRAME_TO
    output_dim = env.action_size

    filters, strides, paddings, hidden_sizes, lr, id_postfix = _create_convnet_options(input_shape)

    main_dqn = DQN(sess, env.id + id_postfix, input_shape, filters, strides, paddings, hidden_sizes, output_dim,
                   learning_rate=lr, name='main')
    target_dqn = DQN(sess, env.id + id_postfix, input_shape, filters, strides, paddings, hidden_sizes, output_dim,
                     learning_rate=lr, name='target', write_tensor_log=False, visualize=False)

    return main_dqn, target_dqn


def _create_convnet_options(input_shape):
    c = input_shape[-1]

    # Used same filter shapes with Atari Deep Reinforcement Learning:
    # https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    # Made a slight tweak to accommodate to smaller frame size.
    filters = [[16, c, 4, 4],
               [32, 16, 2, 2]]
    strides = [2, 1]
    paddings = ['SAME', 'SAME']
    hidden_sizes = [256]

    lr = 5e-4

    conv_repr = '_'.join(['({} {}x{}x{} s_{})'.format(f[0], f[2], f[3], f[1], s) for f, s in zip(filters, strides)])
    hidden_size_repr = 'h_({})'.format(', '.join([str(h) for h in hidden_sizes]))
    frame_size = 'x'.join([str(s) for s in input_shape[:2]])
    id_postfix = ' ({}) {} {}'.format(frame_size, conv_repr, hidden_size_repr)

    return filters, strides, paddings, hidden_sizes, lr, id_postfix


def cleanup(checkpoint, replay_manager, replay_memory_writer, episode, **_):
    checkpoint.save(episode, force=True)
    replay_manager.save(episode, force=True)
    replay_memory_writer.save()


def _train(sess, main_dqn, target_dqn, env, episodes, action_callback, ene_mode):
    logger = Logger(main_dqn.log_dir_path)
    clear_manager = ClearManager()
    replay_manager = ReplayManager(main_dqn.id, flush_frequency=CHECKPOINT_FREQUENCY)
    checkpoint = Checkpoint(sess, main_dqn.id, save_frequency=CHECKPOINT_FREQUENCY).load()
    replay_memory_writer = ReplayMemoryWriter(main_dqn.id).load()

    if env.network_mode == Environment.convolution:
        TensorVisualizer.instance.id = main_dqn.id
        TensorVisualizer.instance.setup()

    CleanupHelper.instance.cleanup = cleanup

    copy_operations = target_dqn.build_copy_variables_from_operation(main_dqn)
    sess.run(copy_operations)

    select = ene.modes[ene_mode]
    possible_states = getattr(env, 'possible_states', None)
    action_spec = {
        'count': env.action_size,
        'generator': lambda s: main_dqn.predict(s)[0],
    }

    for episode in range(checkpoint.next_episode, episodes):
        frame_queue = FrameQueue(FRAME_QUEUE_LENGTH)
        new_frame_queue = FrameQueue(FRAME_QUEUE_LENGTH)

        with replay_manager.managed_random_context(ReplayManager.KEY_RESET):
            state = env.reset()
        done = False
        clear_manager.do_soft_reset()

        Q = main_dqn.predict(possible_states) if possible_states else None
        while not done:
            frame_queue.fill(state)
            action = select(episode, frame_queue.to_states(), action_spec)
            with replay_manager.managed_random_context(action):
                actual_action, new_state, reward, done = env.step(action)
                new_frame_queue.fill(new_state)

            clear_manager.save_reward(reward)
            if frame_queue.is_full:
                states = frame_queue.to_states()
                new_states = new_frame_queue.to_states()
                replay_memory_writer.append((states, action, reward, new_states, done))

            if len(replay_memory_writer) > BATCH_SIZE:
                DQNMixin.replay_train(main_dqn, target_dqn, replay_memory_writer.replay_memory,
                                      DISCOUNT_RATE, minibatch_size=BATCH_SIZE)

                if env.network_mode == Environment.convolution:
                    TensorVisualizer.instance.save_history()
            if env.steps % TARGET_UPDATE_FREQUENCY == 0:
                sess.run(copy_operations)

            state = new_state

            action_callback(env, Q, episode, state, action, actual_action)

        checkpoint.save(episode)
        replay_manager.save(episode)
        if Q is not None:
            summary = env.get_summary_lines(Q)
            logger.log_summary(episode, summary)

        clear_manager.update_last_100_games_rewards()
        clear_manager.print_progress(episode, env.steps)
        if clear_manager.has_cleared(env):
            clear_manager.print_cleared_message(episode)
            simulate_play(env, main_dqn)
            break

        if CleanupHelper.instance.try_to_cleanup(checkpoint, replay_manager, replay_memory_writer, episode):
            break
