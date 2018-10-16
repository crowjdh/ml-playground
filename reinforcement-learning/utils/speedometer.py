from time import time

import numpy as np


class Speedometer:
    def __init__(self):
        self.reset()

    def put(self):
        self.timestamps.append(self.now)

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.timestamps = []

    @property
    def average(self):
        if len(self.timestamps) < 2:
            return None

        return (self.timestamps[-1] - self.timestamps[0]) / (len(self.timestamps) - 1)

    @property
    def proportions(self):
        timestamps = np.asarray(self.timestamps)
        intervals = np.empty_like(timestamps)
        for i in range(1, len(timestamps)):
            intervals[i] = timestamps[i] - timestamps[i-1]
        intervals = intervals[1:]
        return intervals / intervals.sum()

    @property
    def now(self):
        return time()
