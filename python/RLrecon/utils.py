import sys
import time


def print_debug(variable_name):
    frame = sys._getframe(1)
    print("{}={}".format(variable_name, eval(variable_name, frame.f_globals, frame.f_locals)))


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
