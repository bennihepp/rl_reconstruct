from __future__ import print_function

import threading
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging
from pybh import thread_utils


# TODO: Remove or move to support libraries
class QueueBridge(object):
    def __init__(self, coord, dequeue_fn, enqueue_fn, transform_fn=None, verbose=False):
        self._dequeue_fn = dequeue_fn
        self._enqueue_fn = enqueue_fn
        self._coord = coord
        if transform_fn is None:
            def transform_fn(x):
                return x
        self._transform_fn = transform_fn
        self._thread = None
        self._verbose = verbose

    def _run(self):
        while not self._coord.should_stop():
            try:
                entry = self._dequeue_fn()
            except StopIteration:
                print("Input queue closed.")
                break
            if entry == hdf5_utils.QUEUE_END:
                continue
            transformed_entry = self._transform_fn(entry)
            self._enqueue_fn(transformed_entry)
        if self._verbose:
            print("Stop request... Exiting QueueBridge thread")

    def start(self):
        assert (self._thread is None)
        self._thread = threading.Thread(target=self._run, name="QueueBridge")
        self._thread.daemon = True
        self._thread.start()

    @property
    def thread(self):
        return self._thread


# TODO: Remove or move to support libraries
class TFInputPipeline(object):

    def __init__(self, tensor_provider_fn, sess, coord, batch_size,
                 tensor_shapes, tensor_dtypes,
                 queue_capacity, min_after_dequeue,
                 shuffle, num_threads,
                 provides_batches=False,
                 timeout=60, name=None, verbose=False):
        with tf.device("/cpu:0"):
            # Create python-TF data bridge
            assert(queue_capacity > min_after_dequeue)
            self._data_bridge = TFDataBridge(
                sess, batch_size,
                shapes=tensor_shapes,
                dtypes=tensor_dtypes,
                queue_capacity=queue_capacity,
                min_after_dequeue=min_after_dequeue,
                shuffle=shuffle,
                timeout=timeout,
                name=name,
                verbose=verbose)

            if provides_batches:
                enqueue_fn = self._data_bridge.enqueue_batch
            else:
                enqueue_fn = self._data_bridge.enqueue
            # We need an extra thread that dequeues from the multiprocessing queue
            # and enqueues on the threading queue
            self._queue_bridges = [QueueBridge(
                coord,
                tensor_provider_fn,
                enqueue_fn,
                verbose=verbose) for _ in range(num_threads)]

            self._tensor_provider_fn = tensor_provider_fn
            self._enqueue_fn = enqueue_fn

            # Retrieve tensors from data bridge
            self._tensors_batch = self._data_bridge.deque_batch()
            self._tensors = self._data_bridge.deque()

    def start(self):
        for bridge in self._queue_bridges:
            bridge.start()

    @property
    def tensor_provider_fn(self):
        return self._tensor_provider_fn

    @property
    def enqueue_fn(self):
        return self._enqueue_fn

    @property
    def data_bridge(self):
        return self._data_bridge

    @property
    def tensors_batch(self):
        return self._tensors_batch

    @property
    def tensors(self):
        return self._tensors

    @property
    def threads(self):
        return [bridge.thread for bridge in self._queue_bridges]


# TODO: Move to support libraries
class TFBatchFeeder(object):

    def __init__(self, sess, coord, put_ops, group_size, perform_num_ops_on_start=-1):
        self._sess = sess
        self._put_ops = put_ops
        self._group_size = group_size
        if perform_num_ops_on_start >= 0:
            self._perform_num_ops_on_start = perform_num_ops_on_start
        else:
            self._perform_num_ops_on_start = 2 * group_size
        self._num_consumption = 0
        self._thread = thread_utils.CoordinatedThread(coord, target=self._run, name="TFBatchFeeder")
        self._thread.daemon = True
        self._barrier = threading.Barrier(2)

    def _run(self):
        try:
            if self._perform_num_ops_on_start > 0:
                for _ in range(self._perform_num_ops_on_start):
                    self._sess.run(self._put_ops)
            while not self._thread.stopped():
                try:
                    self._barrier.wait()
                    for _ in range(self._group_size):
                        self._sess.run(self._put_ops)
                except threading.BrokenBarrierError as exc:
                    if not self._thread.stopped():
                        raise
        finally:
            self._barrier.abort()

    def start(self):
        self._thread.start()

    def abort(self):
        self._barrier.abort()

    def notify_batch_consumption(self):
        self._num_consumption += 1
        if self._num_consumption % self._group_size == 0:
            self._barrier.wait()

    @property
    def thread(self):
        return self._thread


# TODO: Move to support libraries
class TFStagingArea(object):

    @staticmethod
    def _create_staging_area_from_tensors(tensors):
        return tf_staging.StagingArea(
            dtypes=[tensor.dtype for tensor in tensors],
            shapes=[tensor.shape for tensor in tensors])

    def __init__(self, tensors, device_name=None):
        if device_name is None:
            self._staging_area = self._create_staging_area_from_tensors(tensors)
        else:
            with tf.device(device_name):
                self._staging_area = self._create_staging_area_from_tensors(tensors)
        self._put_op = self._staging_area.put(tensors)
        self._tensors = self._staging_area.get()
        self._size = self._staging_area.size()

    @property
    def put_op(self):
        return self._put_op

    @property
    def tensors(self):
        return self._tensors

    @property
    def size(self):
        return self._size


# TODO: Remove or move to support libraries
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
        if self._verbose:
            print("Stop request... exiting filename queue get next loop")
        raise StopIteration()

    @property
    def thread(self):
        return self._thread


# TODO: Remove or move to support libraries
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
                           name="{}_tensor_{}".format(name, i)) for i in range(len(dtypes))]
        self._tensors_batch = [
            tf.placeholder(dtype=dtypes[i],
                           shape=[None] + list(shapes[i]),
                           name="{}_tensor_batch_{}".format(name, i)) for i in range(len(dtypes))]

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

    @property
    def queue(self):
        return self._queue

    def enqueue(self, tensors):
        assert(len(tensors) == len(self._tensors))
        try:
            self._sess.run([self._enqueue_op],
                           feed_dict={
                               self._tensors[i]: tensors[i] for i in range(len(self._tensors))},
                           options=self._enqueue_options)
            if self._verbose:
                self._enqueue_counter += 1
                if self._enqueue_counter % 100 == 0:
                    print("{} queue: enqueued {}".format(self._name, self._enqueue_counter))
            return True
        except tf.errors.DeadlineExceededError:
            return False

    def enqueue_batch(self, tensors):
        # for tensor in tensors:
        #     assert(tensor.shape[0] == self._batch_size)
        try:
            self._sess.run(self._enqueue_batch_op,
                           feed_dict={
                               self._tensors_batch[i]: tensors[i] for i in range(len(self._tensors))},
                           options=self._enqueue_options)
            return True
        except tf.errors.DeadlineExceededError:
            return False

    def deque(self):
        return self._dequeue_op

    def deque_batch(self):
        return self._dequeue_batch_op
