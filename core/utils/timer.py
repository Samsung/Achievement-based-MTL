import time


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.count = dict()
        self.start_time = dict()
        self.total_time = dict()
        self.epoch_time = time.time()

    def __str__(self):
        return ' '.join(['%s: %3.3f' % (key, time / self.count[key]) for key, time in self.total_time.items()]) + ' sec'

    def tic(self, key='None'):
        if key not in self.count:
            self.count[key] = 0
            self.total_time[key] = 0
        self.start_time[key] = time.time()

    def toc(self, key='None', average=True):
        assert key in self.start_time, 'toc must follow tic'

        diff = time.time() - self.start_time[key]
        self.count[key] += 1
        self.total_time[key] += diff
        return self.total_time[key] / self.count[key] if average else diff

    def get_epoch_time(self):
        epoch_time = time.time() - self.epoch_time
        self.epoch_time = time.time()
        return epoch_time

    def clear(self):
        self.count.clear()
        self.start_time.clear()
        self.total_time.clear()
