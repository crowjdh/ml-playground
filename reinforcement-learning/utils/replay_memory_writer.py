from collections import deque

from utils import numpy_writer
from utils.path import cache_dir_path

REPLAY_MEMORY_FILE_NAME = 'replay_memory'
REPLAY_MEMORY_DIR_NAME = 'replay_memory'


class ReplayMemoryWriter:
    def __init__(self, memory_id):
        self.memory_id = memory_id
        self._replay_memory = None

    def __len__(self):
        return len(self._replay_memory) if self._replay_memory is not None else 0

    def load(self):
        replay_cache = numpy_writer.load_array(self.replay_memory_file_path, REPLAY_MEMORY_FILE_NAME)
        self._replay_memory = deque(replay_cache, maxlen=50000) if replay_cache is not None else deque(maxlen=50000)

        return self

    def append(self, item):
        self._replay_memory.append(item)

    def save(self):
        numpy_writer.save_array(self._replay_memory, self.replay_memory_file_path, REPLAY_MEMORY_FILE_NAME)

    @property
    def replay_memory(self):
        return self._replay_memory

    @property
    def replay_memory_file_path(self):
        return cache_dir_path(self.memory_id, REPLAY_MEMORY_DIR_NAME)
