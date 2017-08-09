from __future__ import print_function

import threading
import Queue
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging
import tf_utils


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
            entry = self._dequeue_fn()
            if entry is None:
                continue
            transformed_entry = self._transform_fn(entry)
            self._enqueue_fn(transformed_entry)
        if self._verbose:
            print("Stop request... Exiting QueueBridge thread")

    def start(self):
        assert (self._thread is None)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    @property
    def thread(self):
        return self._thread


class TFInputPipeline(object):

    def __init__(self, tensor_provider_fn, sess, coord, batch_size,
                 tensor_shapes, tensor_dtypes,
                 queue_capacity, min_after_dequeue,
                 shuffle, num_threads,
                 gpu_device_name=tf_utils.gpu_device_name(),
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

            # We need an extra thread that dequeues from the multiprocessing queue
            # and enqueues on the threading queue
            self._queue_bridges = [QueueBridge(
                coord,
                tensor_provider_fn,
                self._data_bridge.enqueue,
                verbose=verbose) for _ in xrange(num_threads)]

            # Retrieve tensors from data bridge
            self._tensors = self._data_bridge.deque_batch()

    def start(self):
        for bridge in self._queue_bridges:
            bridge.start()

    @property
    def tensors(self):
        return self._tensors

    @property
    def threads(self):
        return [bridge.thread for bridge in self._queue_bridges]


class TFStagingArea(object):

    def __init__(self, tensors, device_name=None):
        if device_name is None:
            self._staging_area = self._create_staging_area(tensors)
        else:
            with tf.device(device_name):
                self._staging_area = self._create_staging_area(tensors)
        self._preload_op = self._staging_area.put(tensors)
        self._tensors = self._staging_area.get()

    def _create_staging_area(self, tensors):
        return tf_staging.StagingArea(
            dtypes=[tensor.dtype for tensor in tensors],
            shapes=[tensor.shape for tensor in tensors])

    @property
    def preload_op(self):
        return self._preload_op

    @property
    def tensors(self):
        return self._tensors


class TFGpuInputPipeline(TFInputPipeline):

    def _gpu_preload_pipeline(self, tensors, gpu_device_name="/gpu:0"):
        with tf.device(gpu_device_name):
            gpu_staging_area = tf_staging.StagingArea(
                dtypes=[tensor.dtype for tensor in tensors],
                shapes=[tensor.shape for tensor in tensors])
        gpu_preload_op = gpu_staging_area.put(tensors)
        gpu_tensors = gpu_staging_area.get()
        return gpu_preload_op, gpu_tensors

    def __init__(self, *args, **kwargs):
        gpu_device_name = kwargs.get("gpu_device_name", tf_utils.gpu_device_name())
        if "gpu_device_name" in kwargs:
            del kwargs["gpu_device_name"]
        super(TFGpuInputPipeline, self).__init__(*args, **kwargs)

        # Generate GPU preload operations
        self._gpu_preload_op, self._tensors = \
            self._gpu_preload_pipeline(self._tensors, gpu_device_name)

    @property
    def gpu_preload_op(self):
        return self._gpu_preload_op


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
