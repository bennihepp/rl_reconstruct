import time


class Timer(object):

    def __init__(self):
        self._t0 = time.time()

    def restart(self):
        t1 = time.time()
        dt = t1 - self._t0
        self._t0 = time.time()
        return dt

    def elapsed_seconds(self):
        dt = time.time() - self._t0
        return dt
