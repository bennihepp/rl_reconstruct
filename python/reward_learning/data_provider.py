from __future__ import print_function

import threading
import Queue
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging


class TFDataProvider(object):

    def __init__(self,
                 epoch_generator_factory,
                 batch_size,
                 input_shape,
                 target_shape,
                 num_epochs=-1,
                 queue_capacity=1024 * 128,
                 gpu_device_name="/gpu:0",
                 name="<default>",
                 verbose=False):
        self._epoch_generator_factory = epoch_generator_factory
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._queue_capacity = queue_capacity
        self._name = name
        self._verbose = verbose

        self._input_in = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._target_in = tf.placeholder(dtype=tf.float32, shape=target_shape)
        self._input_batch_in = tf.placeholder(dtype=tf.float32, shape=[batch_size] + input_shape)
        self._target_batch_in = tf.placeholder(dtype=tf.float32, shape=[batch_size] + target_shape)

        if self._verbose:
            print("Creating FIFO queue with capacity {}: {}".format(queue_capacity, self._name))
        with tf.container("queue-container"):
            self._queue = tf.FIFOQueue(
                capacity=self._queue_capacity,
                dtypes=[tf.float32, tf.float32],
                shapes=[self._input_in.shape, self._target_in.shape])
            # self._queue = tf.RandomShuffleQueue(
            #     capacity=2000,
            #     min_after_dequeue=1000,
            #     dtypes=[tf.float32, tf.float32],
            #     shapes=[self._input_in.shape, self._target_in.shape])
        with tf.device(gpu_device_name):
            self._gpu_staging_area = tf_staging.StagingArea(
                dtypes=[tf.float32, tf.float32],
                shapes=[self._input_batch_in.shape, self._target_batch_in.shape])

        self._enqueue_op = self._queue.enqueue([self._input_in, self._target_in])
        self._enqueue_batch_op = self._queue.enqueue_many([self._input_batch_in, self._target_batch_in])

        self._queue_batch_counts = []
        self._queue_epoch_complete = []
        self._epoch = 0

        self._stop = False
        self._thread = None
        self._lock = threading.Lock()

    def _run(self, sess):
        """Thread loop. Feeds new data into queue."""
        def enqueue_epoch():
            epoch_generator = self._epoch_generator_factory()
            with self._lock:
                self._queue_batch_counts.append(0)
                self._queue_epoch_complete.append(False)
            for input_batch, target_batch in epoch_generator:
                assert(input_batch.shape[0] == self._batch_size)
                assert(target_batch.shape[0] == self._batch_size)
                sess.run(self._enqueue_batch_op, feed_dict={
                    self._input_batch_in: input_batch,
                    self._target_batch_in: target_batch})
                # for j in xrange(record_batch.rewards.shape[0]):
                #     sess.run(self._enqueue_op, feed_dict={
                #         self._rewards_in: target_batch[j, ...],
                #         self._input_in: input_batch[j, ...]})
                with self._lock:
                    self._queue_batch_counts[-1] += 1
                    # assert(np.sum(self._queue_batch_counts) <= self._queue_capacity)
                if self._verbose:
                    print(self._name, "queue_batch_counts:", self._queue_batch_counts)
            with self._lock:
                self._queue_epoch_complete[-1] = True
        if self._num_epochs <= 0:
            while not self._stop:
                enqueue_epoch()
        else:
            for i in xrange(self._num_epochs):
                enqueue_epoch()
                if self._stop:
                    break

    def start_thread(self, sess):
        """Start thread to fill queue """
        assert(self._thread is None)
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(sess,))
        self._thread.daemon = True  # Kill thread when all other threads stopped
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop = True

    def join_thread(self):
        self._thread.join()
        self._thread = None

    def _deque_batch(self):
        return self._queue.dequeue_many(self._batch_size)

    def next_batch(self):
        return self._gpu_staging_area.get()

    def consume_batch(self):
        with self._lock:
            assert(self._queue_batch_counts[0] > 0)
            self._queue_batch_counts[0] -= 1
            if self._verbose:
                print(self._name, "queue_batch_counts:", self._queue_batch_counts)
            if self._queue_epoch_complete[0] and self._queue_batch_counts[0] == 0:
                self._queue_epoch_complete[:1] = []
                self._queue_batch_counts[:1] = []
                self._epoch += 1
                return True
            return False

    def get_epoch(self):
        return self._epoch

    # def get_batch_count(self, sess):
    #     return self._batch_count
    #
    # def get_record_count(self, sess):
    #     return self._record_count

    def preload_gpu_op(self):
        input_batch, target_batch = self._deque_batch()
        return self._gpu_staging_area.put([input_batch, target_batch])


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

    def enqueue(self, tensors):
        assert(len(tensors) == len(self._tensors))
        try:
            self._sess.run([self._enqueue_op],
                           feed_dict={
                               self._tensors[i]: tensors[i] for i in xrange(len(self._tensors))},
                           options=self._enqueue_options)
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


class TFDataProvider(object):

    def __init__(self,
                 epoch_generator_factory,
                 batch_size,
                 input_shape,
                 target_shape,
                 num_epochs=-1,
                 queue_capacity=1024 * 128,
                 gpu_device_name="/gpu:0",
                 name="<default>",
                 verbose=False):
        self._epoch_generator_factory = epoch_generator_factory
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._queue_capacity = queue_capacity
        self._name = name
        self._verbose = verbose

        self._input_in = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self._target_in = tf.placeholder(dtype=tf.float32, shape=target_shape)
        self._input_batch_in = tf.placeholder(dtype=tf.float32, shape=[batch_size] + input_shape)
        self._target_batch_in = tf.placeholder(dtype=tf.float32, shape=[batch_size] + target_shape)

        if self._verbose:
            print("Creating FIFO queue with capacity {}: {}".format(queue_capacity, self._name))
        with tf.container("queue-container"):
            self._queue = tf.FIFOQueue(
                capacity=self._queue_capacity,
                dtypes=[tf.float32, tf.float32],
                shapes=[self._input_in.shape, self._target_in.shape])
            # self._queue = tf.RandomShuffleQueue(
            #     capacity=2000,
            #     min_after_dequeue=1000,
            #     dtypes=[tf.float32, tf.float32],
            #     shapes=[self._input_in.shape, self._target_in.shape])
        with tf.device(gpu_device_name):
            self._gpu_staging_area = tf_staging.StagingArea(
                dtypes=[tf.float32, tf.float32],
                shapes=[self._input_batch_in.shape, self._target_batch_in.shape])

        self._enqueue_op = self._queue.enqueue([self._input_in, self._target_in])
        self._enqueue_batch_op = self._queue.enqueue_many([self._input_batch_in, self._target_batch_in])

        self._queue_batch_counts = []
        self._queue_epoch_complete = []
        self._epoch = 0

        self._stop = False
        self._thread = None
        self._lock = threading.Lock()

    def _run(self, sess):
        """Thread loop. Feeds new data into queue."""

        def enqueue_epoch():
            epoch_generator = self._epoch_generator_factory()
            with self._lock:
                self._queue_batch_counts.append(0)
                self._queue_epoch_complete.append(False)
            for input_batch, target_batch in epoch_generator:
                assert (input_batch.shape[0] == self._batch_size)
                assert (target_batch.shape[0] == self._batch_size)
                sess.run(self._enqueue_batch_op, feed_dict={
                    self._input_batch_in: input_batch,
                    self._target_batch_in: target_batch})
                # for j in xrange(record_batch.rewards.shape[0]):
                #     sess.run(self._enqueue_op, feed_dict={
                #         self._rewards_in: target_batch[j, ...],
                #         self._input_in: input_batch[j, ...]})
                with self._lock:
                    self._queue_batch_counts[-1] += 1
                    # assert(np.sum(self._queue_batch_counts) <= self._queue_capacity)
                if self._verbose:
                    print(self._name, "queue_batch_counts:", self._queue_batch_counts)
            with self._lock:
                self._queue_epoch_complete[-1] = True

        if self._num_epochs <= 0:
            while not self._stop:
                enqueue_epoch()
        else:
            for i in xrange(self._num_epochs):
                enqueue_epoch()
                if self._stop:
                    break

    def start_thread(self, sess):
        """Start thread to fill queue """
        assert (self._thread is None)
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(sess,))
        self._thread.daemon = True  # Kill thread when all other threads stopped
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop = True

    def join_thread(self):
        self._thread.join()
        self._thread = None

    def _deque_batch(self):
        return self._queue.dequeue_many(self._batch_size)

    def next_batch(self):
        return self._gpu_staging_area.get()

    def consume_batch(self):
        with self._lock:
            assert (self._queue_batch_counts[0] > 0)
            self._queue_batch_counts[0] -= 1
            if self._verbose:
                print(self._name, "queue_batch_counts:", self._queue_batch_counts)
            if self._queue_epoch_complete[0] and self._queue_batch_counts[0] == 0:
                self._queue_epoch_complete[:1] = []
                self._queue_batch_counts[:1] = []
                self._epoch += 1
                return True
            return False

    def get_epoch(self):
        return self._epoch

    # def get_batch_count(self, sess):
    #     return self._batch_count
    #
    # def get_record_count(self, sess):
    #     return self._record_count

    def preload_gpu_op(self):
        input_batch, target_batch = self._deque_batch()
        return self._gpu_staging_area.put([input_batch, target_batch])


# TODO: Update
class DataProvider(object):

    def __init__(self, epoch_generator_factory,
                 num_epochs=-1):
        self._epoch_generator_factory = epoch_generator_factory
        self._num_epochs = num_epochs

        self._batch_queue = Queue.Queue(20000)
        self._epoch = 0
        self._batch_count = 0
        self._record_count = 0

        self._thread = None

    def _run(self):
        """Thread loop. Feeds new data into queue."""
        def enqueue_epoch():
            epoch_generator = self._epoch_generator_factory()
            for record_batch in epoch_generator:
                self._batch_queue.put(record_batch)
            self._batch_queue.put(None)
        if self._num_epochs <= 0:
            while True:
                enqueue_epoch()
        else:
            for i in xrange(self._num_epochs):
                enqueue_epoch()

    def start_thread(self):
        """Start thread to fill queue """
        assert(self._thread is None)
        self._batch_count = 0
        self._record_count = 0
        self._thread = threading.Thread(target=self._run)
        self._thread.daemon = True  # Kill thread when all other threads stopped
        self._thread.start()
        return self._thread

    def join_thread(self):
        self._thread.join()
        self._thread = None

    def next_batch(self):
        record_batch = self._batch_queue.get()
        if record_batch is None:
            self._epoch += 1
            return None
        self._batch_count += 1
        self._record_count += record_batch.grid_3ds.shape[0]
        return record_batch

    def get_queue_size(self):
        return self._batch_queue.qsize()

    def get_epoch(self):
        return self._epoch

    def get_batch_count(self):
        return self._batch_count

    def get_record_count(self):
        return self._record_count
