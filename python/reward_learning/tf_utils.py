from __future__ import print_function

import threading
import Queue
import numpy as np
import tensorflow as tf


class FilenameQueueProvider(object):
    def __init__(self, filenames, coord, shuffle, timeout=60, verbose=False):
        self._coord = coord
        self._shuffle = shuffle
        self._timeout = timeout
        self._verbose = verbose
        # Make copy
        self._filenames = list(filenames)
        self._filename_queue = Queue.Queue(maxsize=2 * len(self._filenames))
        self._epoch = 0
        self._thread = None

    def _run(self):
        while not self._coord.should_stop():
            if self._shuffle:
                np.random.shuffle(self._filenames)
            for filename in self._filenames:
                entry = (filename, self._epoch)
                enqueued = False
                while not enqueued:
                    try:
                        self._filename_queue.put(entry, block=True, timeout=self._timeout)
                        enqueued = True
                    except Queue.Full:
                        pass
                    if self._coord.should_stop():
                        break
                if self._coord.should_stop():
                    break
            self._epoch += 1
        if self._verbose:
            print("Stop request... Exiting filename queue thread")

    def start(self):
        assert (self._thread is None)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def get_next(self):
        while not self._coord.should_stop():
            try:
                return self._filename_queue.get(block=True, timeout=self._timeout)
            except Queue.Empty:
                pass
        print("Stop request... exiting filename queue get next loop")
        raise StopIteration()

    @property
    def thread(self):
        return self._thread


def get_activation_function_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "relu":
        return tf.nn.relu
    elif name == "elu":
        return tf.nn.elu
    elif name == "tanh":
        return tf.nn.tanh
    elif name == "sigmoid":
        return tf.nn.sigmoid
    elif name == "none":
        return None
    else:
        raise NotImplementedError("Unknown activation function: {}".format(name))


def get_optimizer_by_name(name, default):
    if name is None:
        return default
    name = name.lower()
    if name == "adam":
        return tf.train.AdamOptimizer
    elif name == "sgd":
        return tf.train.GradientDescentOptimizer
    elif name == "rmsprop":
        return tf.train.RMSPropOptimizer
    elif name == "adadelta":
        return tf.train.AdadeltaOptimizer
    elif name == "momentum":
        return tf.train.MomentumOptimizer
    elif name == "adagrad":
        return tf.train.AdagradOptimizer
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(name))
