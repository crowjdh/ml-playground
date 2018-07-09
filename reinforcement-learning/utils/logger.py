import os
from collections import Iterable


class Logger:
    def __init__(self, name):
        self.name = '{}.txt'.format(name)
        if os.path.isfile(self.name):
            os.remove(self.name)

    def log(self, lines):
        if isinstance(lines, str) or not isinstance(lines, Iterable):
            lines = [lines]
        with open(self.name, 'a') as log_file:
            for line in lines:
                log_file.write(str(line) + '\n')

    def log_summary(self, episode, summary):
        self.log('Episode {}\n'.format(episode))
        self.log(summary)
