from time import time


class Speedometer:
    def __init__(self):
        self.reset()

    def put(self):
        self.timestamps.append(self.now)

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.timestamps = [self.now]

    @property
    def average(self):
        if len(self.timestamps) < 2:
            return None

        return (self.timestamps[-1] - self.timestamps[0]) / (len(self.timestamps) - 1)

    @property
    def now(self):
        return time()
