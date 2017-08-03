from __future__ import print_function

import threading
import Queue
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging


class TFDataBridge(object):

    def __init__(self,
                 sess,
                 batch_size,
                 shapes,
                 dtypes,
                 queue_capacity=1024 * 128,
                 min_after_dequeue=1000,
                 shuffle=True,
                 name="<default>",
                 verbose=False,
                 timeout=60):
        self._sess = sess
        self._batch_size = batch_size
        self._queue_capacity = queue_capacity
        self._min_after_dequeue = min_after_dequeue
        self._name = name
        self._verbose = verbose

        assert(len(shapes) == len(dtypes))
        self._shapes = shapes
        self._dtypes = dtypes
        self._tensors = [
            tf.placeholder(dtype=dtypes[i],
                           shape=shapes[i],
                           name="{}_tensor_{}".format(name, i)) for i in xrange(len(dtypes))]
        self._tensors_batch = [
            tf.placeholder(dtype=dtypes[i],
                           shape=[batch_size] + list(shapes[i]),
                           name="{}_tensor_batch_{}".format(name, i)) for i in xrange(len(dtypes))]

        if self._verbose:
            print("Creating queue with capacity {}: {}".format(queue_capacity, self._name))
        if shuffle:
            self._queue = tf.RandomShuffleQueue(
                capacity=self._queue_capacity,
                min_after_dequeue=self._min_after_dequeue,
                dtypes=self._dtypes,
                shapes=self._shapes)
        else:
            self._queue = tf.FIFOQueue(
                capacity=self._queue_capacity,
                dtypes=self._dtypes,
                shapes=self._shapes)

        self._enqueue_op = self._queue.enqueue(self._tensors)
        self._enqueue_batch_op = self._queue.enqueue_many(self._tensors_batch)
        self._dequeue_op = self._queue.dequeue()
        self._dequeue_batch_op = self._queue.dequeue_many(self._batch_size)

        self._enqueue_options = tf.RunOptions(timeout_in_ms=timeout * 1000)
        self._enqueue_counter = 0

    def enqueue(self, tensors):
        assert(len(tensors) == len(self._tensors))
        try:
            self._sess.run([self._enqueue_op],
                           feed_dict={
                               self._tensors[i]: tensors[i] for i in xrange(len(self._tensors))},
                           options=self._enqueue_options)
            if self._verbose:
                self._enqueue_counter += 1
                if self._enqueue_counter % 100 == 0:
                    print("{} queue: enqueued {}".format(self._name, self._enqueue_counter))
            return True
        except tf.errors.DeadlineExceededError:
            return False

    def enqueue_batch(self, tensors):
        for tensor in tensors:
            assert(tensor.shape[0] == self._batch_size)
        try:
            self._sess.run(self._enqueue_batch_op,
                           feed_dict={self._tensors: tensors},
                           options=self._enqueue_options)
            return True
        except tf.errors.DeadlineExceededError:
            return False

    def deque(self):
        return self._dequeue_op

    def deque_batch(self):
        return self._dequeue_batch_op
