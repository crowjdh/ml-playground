import os
from shutil import copyfile
from contextlib import contextmanager
import pathlib

import numpy as np

from utils.path import cache_dir_path

REPLAY_ROOT_DIR_PATH = 'replay'


def _parse_episode_range(episodes, start_idx, end_idx):
    start_idx = start_idx if start_idx is not None else 1
    end_idx = end_idx if end_idx is not None else episodes

    assert 1 <= start_idx <= end_idx <= episodes, "Invalid index were given. Must satisfy: " \
                                                  "1 <= start_idx <= end_idx <= episodes({})\n" \
                                                  "Given: " \
                                                  "start_idx: {}, end_idx: {}".format(episodes, start_idx, end_idx)

    return start_idx, end_idx


# TODO: Replace logic for saving/loading numpy array with numpy_writer.py
class ReplayManager:
    KEY_RESET = -1

    def __init__(self, replay_id, flush_frequency=10):
        self.id = replay_id
        self.flush_frequency = flush_frequency
        self._reset_buffers()
        self._reset_buffer()

    @contextmanager
    def managed_random_context(self, action):
        seed = np.random.randint(0, 2**32)
        state = np.random.get_state()

        try:
            np.random.seed(seed)
            yield
            self.buffer.append((seed, action))
        finally:
            np.random.set_state(state)

    @contextmanager
    def context(self, seed):
        state = np.random.get_state()

        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)

    def save(self, episode, force=False):
        self._pack_buffer()
        if episode % self.flush_frequency == 0 or force:
            self._flush_as(episode)

    def replay(self, env, start_idx=None, end_idx=None):
        episodes, history = self._load()

        if not episodes or not history:
            raise IOError("Either file {} or {} does not exists."
                          .format(self.state_file_path, self.history_file_path))

        start_idx, end_idx = _parse_episode_range(episodes, start_idx, end_idx)

        for episode in range(start_idx, end_idx + 1):
            seed_pairs = history[episode - 1]
            reset_seed_pair = seed_pairs[0]
            seed_action_pairs = seed_pairs[1:]

            seed, code = reset_seed_pair
            with self.context(seed):
                env.reset()

            for seed_action_pair in seed_action_pairs:
                seed, action = seed_action_pair
                with self.context(seed):
                    env.step(action)
            print("Episode: {:5.0f}, steps: {:5.0f}".format(episode, len(seed_action_pairs)))

    def _load(self):
        episodes = self._load_episode()
        history = self._load_history(episodes)

        return episodes, history

    def _load_episode(self):
        if not os.path.isfile(self.state_file_path):
            return None

        with open(self.state_file_path, 'r') as state_file:
            return int(state_file.readline())

    def _load_history(self, episodes):
        if not os.path.isfile(self.history_file_path):
            return None

        with open(self.history_file_path, 'rb') as history_file:
            history = []
            for episode in range(episodes):
                history.append(np.load(history_file))
        return history

    def _pack_buffer(self):
        if len(self.buffer) == 0:
            return
        self.buffers.append(self.buffer)
        self._reset_buffer()

    def _flush_as(self, episode):
        if not self.is_dirty:
            return

        prev_episode = self._load_episode() or 0
        buffer_size = len(self.buffers)
        if episode - prev_episode != buffer_size:
            raise ValueError("")
        self._prepare_tmp_history_file()

        self._save_last_episode_into_temp_file(episode)
        self._append_buffers_into_tmp_file()

        self._replace_with_tmp_files()

        self._reset_buffer()
        self._reset_buffers()

    def _prepare_tmp_history_file(self):
        pathlib.Path(self.replay_dir_path).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(self.history_file_path):
            copyfile(self.history_file_path, self.tmp_history_file_path)

    def _save_last_episode_into_temp_file(self, last_episode):
        with open(self.tmp_state_file_path, 'w') as temp_state_file:
            temp_state_file.write(str(last_episode))

    # noinspection PyTypeChecker
    def _append_buffers_into_tmp_file(self):
        with open(self.tmp_history_file_path, 'ab') as temp_history_file:
            for buffer in self.buffers:
                np.save(temp_history_file, np.asarray(buffer))

    def _replace_with_tmp_files(self):
        os.rename(self.tmp_state_file_path, self.state_file_path)
        os.rename(self.tmp_history_file_path, self.history_file_path)

    # noinspection PyAttributeOutsideInit
    def _reset_buffers(self):
        self.buffers = []

    # noinspection PyAttributeOutsideInit
    def _reset_buffer(self):
        self.buffer = []

    @property
    def is_dirty(self):
        return len(self.buffer) > 0 or len(self.buffers) > 0

    @property
    def replay_dir_path(self):
        return cache_dir_path(self.id, REPLAY_ROOT_DIR_PATH)

    @property
    def state_file_path(self):
        return os.path.join(self.replay_dir_path, 'state')

    @property
    def tmp_state_file_path(self):
        return os.path.join(self.replay_dir_path, 'state.tmp')

    @property
    def history_file_path(self):
        return os.path.join(self.replay_dir_path, 'history')

    @property
    def tmp_history_file_path(self):
        return os.path.join(self.replay_dir_path, 'history.tmp')
