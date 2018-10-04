import os
import pathlib

import tensorflow as tf

CHECKPOINT_DIR_NAME = '.checkpoints'


class Checkpoint:
    def __init__(self, sess, checkpoint_name, save_frequency=10):
        self.sess = sess
        self.checkpoint_name = checkpoint_name
        self.save_frequency = save_frequency
        self.episode_tensor = tf.Variable(initial_value=0, trainable=False)
        self.saver = tf.train.Saver(save_relative_paths=True)
        self.checkpoint_dir_path = os.path.join(CHECKPOINT_DIR_NAME, checkpoint_name)
        self.checkpoint_file = os.path.join(self.checkpoint_dir_path, 'model.ckpt')

        self._ensure_checkpoint_dir()
        self._init_tensors()

    def _ensure_checkpoint_dir(self):
        pathlib.Path(self.checkpoint_dir_path).mkdir(parents=True, exist_ok=True)

    def _init_tensors(self):
        self.sess.run(self.episode_tensor.initializer)

    @property
    def next_episode(self):
        return self.sess.run(self.episode_tensor) + 1

    def save(self, episode):
        if episode % self.save_frequency == 0:
            self.sess.run(self.episode_tensor.assign(episode))
            self.saver.save(self.sess, self.checkpoint_file)

    def load(self):
        checkpoint_state = tf.train.get_checkpoint_state(self.checkpoint_dir_path)
        if checkpoint_state and checkpoint_state.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint_state.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir_path, checkpoint_name))

        return self
