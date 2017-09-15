import sys
import time


def print_debug(variable_name):
    frame = sys._getframe(1)
    print("{}={}".format(variable_name, eval(variable_name, frame.f_globals, frame.f_locals)))


class Timer(object):

    def __init__(self):
        self._t0 = self.total_seconds()

    def restart(self):
        t1 = self.total_seconds()
        dt = t1 - self._t0
        self._t0 = self.total_seconds()
        return dt

    def elapsed_seconds(self):
        dt = self.total_seconds() - self._t0
        return dt

    def seconds(self):
        return self.elapsed_seconds()

    @staticmethod
    def total_seconds():
        return time.time()

    def start_seconds(self):
        return self._t0
