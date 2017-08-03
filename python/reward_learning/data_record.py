from __future__ import print_function

from collections import namedtuple
import threading
import multiprocessing
import Queue
import numpy as np
import h5py
import tensorflow as tf


Record = namedtuple("Record", ["obs_levels", "grid_3d", "rewards", "prob_rewards", "scores"])
RecordBatch = namedtuple("RecordBatch", ["obs_levels", "grid_3ds", "rewards", "prob_rewards", "scores"])

RecordV2 = namedtuple("RecordV2", ["obs_levels", "grid_3d", "rewards", "norm_rewards",
                                 "prob_rewards", "norm_prob_rewards", "scores"])
RecordV2Batch = namedtuple("RecordV2Batch", ["obs_levels", "grid_3ds", "rewards", "norm_rewards",
                                           "prob_rewards", "norm_prob_rewards", "scores"])


def write_hdf5_file(filename, numpy_dict):
    f = h5py.File(filename, "w")
    for key, array in numpy_dict.iteritems():
        array = np.asarray(array)
        if array.dtype == np.float32:
            hdf5_dtype = 'f'
        elif array.dtype == np.float64:
            hdf5_dtype = 'd'
        elif array.dtype == np.int:
            hdf5_dtype = 'i'
        else:
            raise NotImplementedError("Unsupported datatype for write_hdf5_file() helper")
        dataset = f.create_dataset(key, array.shape, dtype=hdf5_dtype)
        dataset[...] = array
    f.close()


def read_hdf5_file_to_numpy_dict(filename):
    f = h5py.File(filename, "r")
    numpy_dict = {}
    for key in f:
        numpy_dict[key] = np.array(f[key])
    f.close()
    return numpy_dict


def write_hdf5_records(filename, records):
    f = h5py.File(filename, "w")
    (obs_levels, grid_3d, rewards, prob_rewards, scores) = records[0]
    assert(grid_3d.shape[-1] == 2 * len(obs_levels))
    rewards_shape = (len(records),) + rewards.shape
    rewards_dset = f.create_dataset("rewards", rewards_shape, dtype='f')
    prob_rewards_shape = (len(records),) + prob_rewards.shape
    prob_rewards_dset = f.create_dataset("prob_rewards", prob_rewards_shape, dtype='f')
    scores_shape = (len(records),) + scores.shape
    scores_dset = f.create_dataset("scores", scores_shape, dtype='f')
    grid_3ds_shape = (len(records),) + grid_3d.shape
    grid_3ds_dset = f.create_dataset("grid_3ds", grid_3ds_shape, dtype='f')
    grid_3ds_dset.attrs["obs_levels"] = obs_levels
    grid_3ds_dset.attrs["obs_channels"] = grid_3ds_shape[-1] / len(obs_levels)
    for i, record in enumerate(records):
        (obs_levels, grid_3d, rewards, prob_rewards, scores) = record
        rewards_dset[i, ...] = rewards
        prob_rewards_dset[i, ...] = prob_rewards
        scores_dset[i, ...] = scores
        grid_3ds_dset[i, ...] = grid_3d
    f.close()


def read_hdf5_records(filename):
    f = h5py.File(filename, "r")
    obs_levels = f["grid_3ds"].attrs["obs_levels"]
    obs_channels = f["grid_3ds"].attrs["obs_channels"]
    rewards = np.array(f["rewards"])
    prob_rewards = np.array(f["prob_rewards"])
    scores = np.array(f["scores"])
    grid_3ds = np.array(f["grid_3ds"])
    assert(rewards.shape[0] == grid_3ds.shape[0])
    assert(prob_rewards.shape[0] == grid_3ds.shape[0])
    assert(scores.shape[0] == grid_3ds.shape[0])
    assert(grid_3ds.shape[-1] == len(obs_levels) * obs_channels)
    return RecordBatch(obs_levels, grid_3ds, rewards, prob_rewards, scores)


def write_hdf5_records_v2(filename, records):
    f = h5py.File(filename, "w")
    (obs_levels, grid_3d, rewards, norm_rewards, prob_rewards, norm_prob_rewards, scores) = records[0]
    assert(grid_3d.shape[-1] == 2 * len(obs_levels))
    rewards_shape = (len(records),) + rewards.shape
    rewards_dset = f.create_dataset("rewards", rewards_shape, dtype='f')
    norm_rewards_shape = (len(records),) + norm_rewards.shape
    norm_rewards_dset = f.create_dataset("norm_rewards", norm_rewards_shape, dtype='f')
    prob_rewards_shape = (len(records),) + prob_rewards.shape
    prob_rewards_dset = f.create_dataset("prob_rewards", prob_rewards_shape, dtype='f')
    norm_prob_rewards_shape = (len(records),) + norm_prob_rewards.shape
    norm_prob_rewards_dset = f.create_dataset("norm_prob_rewards", norm_prob_rewards_shape, dtype='f')
    scores_shape = (len(records),) + scores.shape
    scores_dset = f.create_dataset("scores", scores_shape, dtype='f')
    grid_3ds_shape = (len(records),) + grid_3d.shape
    grid_3ds_dset = f.create_dataset("grid_3ds", grid_3ds_shape, dtype='f')
    grid_3ds_dset.attrs["obs_levels"] = obs_levels
    grid_3ds_dset.attrs["obs_channels"] = grid_3ds_shape[-1] / len(obs_levels)
    for i, record in enumerate(records):
        (obs_levels, grid_3d, rewards, norm_rewards, prob_rewards, norm_prob_rewards, scores) = record
        rewards_dset[i, ...] = rewards
        norm_rewards_dset[i, ...] = norm_rewards
        prob_rewards_dset[i, ...] = prob_rewards
        norm_prob_rewards_dset[i, ...] = norm_prob_rewards
        scores_dset[i, ...] = scores
        grid_3ds_dset[i, ...] = grid_3d
    f.close()


def read_hdf5_records_v2(filename):
    f = h5py.File(filename, "r")
    obs_levels = np.array(f["grid_3ds"].attrs["obs_levels"])
    obs_channels = np.array(f["grid_3ds"].attrs["obs_channels"])
    rewards = np.array(f["rewards"])
    norm_rewards = np.array(f["norm_rewards"])
    prob_rewards = np.array(f["prob_rewards"])
    norm_prob_rewards = np.array(f["norm_prob_rewards"])
    scores = np.array(f["scores"])
    grid_3ds = np.array(f["grid_3ds"])
    f.close()
    assert(rewards.shape[0] == grid_3ds.shape[0])
    assert(norm_rewards.shape[0] == grid_3ds.shape[0])
    assert(prob_rewards.shape[0] == grid_3ds.shape[0])
    assert(norm_prob_rewards.shape[0] == grid_3ds.shape[0])
    assert(scores.shape[0] == grid_3ds.shape[0])
    assert(grid_3ds.shape[-1] == len(obs_levels) * obs_channels)
    return RecordV2Batch(obs_levels, grid_3ds, rewards, norm_rewards, prob_rewards, norm_prob_rewards, scores)


def generate_single_records_from_batch_v2(record_batch):
    obs_levels, grid_3ds, rewards, norm_rewards, prob_rewards, norm_prob_rewards, scores = record_batch
    for i in xrange(rewards.shape[0]):
        obs_levels = np.array(obs_levels)
        single_rewards = rewards[i, ...]
        single_prob_rewards = prob_rewards[i, ...]
        single_norm_rewards = norm_rewards[i, ...]
        single_norm_prob_rewards = norm_prob_rewards[i, ...]
        single_scores = scores[i, ...]
        grid_3d = grid_3ds[i, ...]
        record = RecordV2(obs_levels, grid_3d, single_rewards, single_norm_rewards,
                          single_prob_rewards, single_norm_prob_rewards, single_scores)
        yield record


def read_hdf5_records_as_list(filename):
    obs_levels, grid_3ds, rewards, prob_rewards, scores = read_hdf5_records(filename)
    records = []
    for i in xrange(rewards.shape[0]):
        obs_levels = np.array(obs_levels)
        single_rewards = rewards[i, ...]
        single_prob_rewards = prob_rewards[i, ...]
        single_scores = scores[i, ...]
        grid_3d = grid_3ds[i, ...]
        record = Record(obs_levels, grid_3d, single_rewards, single_prob_rewards, single_scores)
        records.append(record)
    return records


def read_hdf5_records_v2_as_list(filename):
    record_batch = read_hdf5_records_v2(filename)
    records = [record for record in generate_single_records_from_batch_v2(record_batch)]
    return records


def count_records_in_hdf5_file_v2(filename):
    record_batch = read_hdf5_records_v2(filename)
    return record_batch.grid_3ds.shape[0]


class HDF5QueueReader(object):

    def __init__(self, filename_dequeue_fn, enqueue_record_fn, coord, verbose=False):
        self._filename_dequeue_fn = filename_dequeue_fn
        self._enqueue_record_fn = enqueue_record_fn
        self._coord = coord
        self._thread = None
        self._verbose = verbose

    def _run(self):
        record_count = 0
        while not self._coord.should_stop():
            filename = self._filename_dequeue_fn()
            if filename is None:
                continue
            record_batch = read_hdf5_records_v2(filename)
            for record in generate_single_records_from_batch_v2(record_batch):
                enqueued = False
                while not enqueued:
                    enqueued = self._enqueue_record_fn(record)
                    if self._coord.should_stop():
                        break
                if self._coord.should_stop():
                    break
                record_count += 1
            if self._verbose:
                print("Enqueued {} records.".format(record_count))
        if self._verbose:
            print("Stop request... Exiting HDF5 queue reader thread")

    def start(self):
        assert (self._thread is None)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    @property
    def thread(self):
        return self._thread


class HDF5ReaderProcess(object):

    def __init__(self, filename_queue, record_queue, verbose=False):
        self._filename_queue = filename_queue
        self._record_queue = record_queue
        self._process = None
        self._verbose = verbose
        self._should_stop = True

    def _run(self):
        record_count = 0
        while True:
            filename = self._filename_queue.get(block=True)
            if filename is None:
                break
            record_batch = read_hdf5_records_v2(filename)
            for record in generate_single_records_from_batch_v2(record_batch):
                self._record_queue.put(record, block=True)
                record_count += 1
            if self._verbose:
                print("Enqueued {} records.".format(record_count))
        if self._verbose:
            print("Stop request... Exiting HDF5 queue reader process")

    def start(self):
        assert (self._process is None)
        self._should_stop = False
        self._process = multiprocessing.Process(target=self._run)
        self._process.start()

    def is_alive(self):
        if self._process is None:
            return False
        else:
            return self._process.is_alive()

    def join(self):
        if self._process is not None:
            self._process.join()

    @property
    def process(self):
        return self._process


class HDF5ReaderProcessCoordinator(object):

    def __init__(self, filenames, coord, shuffle, timeout=60, num_processes=1,
                 record_queue_capacity=1024, verbose=False):
        self._filenames = list(filenames)
        self._coord = coord
        self._shuffle = shuffle
        self._timeout = timeout
        self._num_processes = num_processes
        self._record_queue_capacity = record_queue_capacity
        self._verbose = verbose

        self._thread = None

        self._initialize_queue_and_readers()

    def _initialize_queue_and_readers(self):
        self._record_queue = multiprocessing.Queue(maxsize=self._record_queue_capacity)
        self._filename_queue = multiprocessing.Queue(maxsize=2 * len(self._filenames))
        self._reader_processes = [
            HDF5ReaderProcess(self._filename_queue, self._record_queue, self._verbose)
            for _ in xrange(self._num_processes)]

    def _start_readers(self):
        for reader in self._reader_processes:
            assert(not reader.is_alive())
            reader.start()

    def _stop_readers(self):
        self._record_queue.close()
        self._filename_queue.close()
        self._record_queue.join_thread()
        self._filename_queue.join_thread()
        if self._verbose:
            print("Terminating HDF5ReaderProcesses to stop")
        for reader in self._reader_processes:
            reader.process.terminate()
        if self._verbose:
            print("Resetting queue and HDF5ReaderProcesses")
        self._initialize_queue_and_readers()

    def _run(self):
        while not self._coord.should_stop():
            if self._shuffle:
                np.random.shuffle(self._filenames)
            for filename in self._filenames:
                enqueued = False
                while not enqueued:
                    try:
                        self._filename_queue.put(filename, block=True, timeout=self._timeout)
                        enqueued = True
                    except Queue.Full:
                        pass
                    if self._coord.should_stop():
                        break
                if self._coord.should_stop():
                    break
        # Terminate reader processes
        self._stop_readers()
        if self._verbose:
            print("Stop request... Exiting HDF5ReaderProcessCoordinator queue thread")

    def get_next_record(self):
        while not self._coord.should_stop():
            try:
                return self._record_queue.get(block=True, timeout=self._timeout)
            except Queue.Empty:
                pass
        if self._verbose:
            print("Stop request... Exiting HDF5ReaderProcessCoordinator queue thread")
        raise StopIteration()

    def start(self):
        assert (self._thread is None)
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        self._start_readers()

    def compute_num_records(self):
        p = multiprocessing.Pool(self._num_processes)
        # TODO: This blocks on Ctrl-C
        try:
            record_counts_async = p.map_async(count_records_in_hdf5_file_v2, self._filenames, chunksize=1)
            # Workaround to enable KeyboardInterrupt (with p.map() or record_counts_async.get() it won't be received)
            record_counts = record_counts_async.get(np.iinfo(np.int32).max)
            record_count = np.sum(record_counts)
            p.close()
            p.join()
        except Exception, exc:
            print("Exception occured when computing num records: {}".format(exc))
            p.close()
            p.terminate()
            raise
        return record_count

    @property
    def thread(self):
        return self._thread


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_array_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float32_array_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _bytes_feature(array):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[array]))


def read_and_decode_tfrecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    int64_features = [
        'obs_levels',
        'grid_3ds_shape',
    ]
    float32_features = [
        # 'grid_3ds',
        # 'rewards',
        # 'prob_rewards',
        # 'norm_rewards',
        # 'norm_prob_rewards',
        # 'scores',
    ]
    bytes_features = [
        'grid_3ds',
        'rewards',
        'prob_rewards',
        'norm_rewards',
        'norm_prob_rewards',
        'scores',
    ]
    features_dict = {}
    features_dict.update({
        name: tf.VarLenFeature(tf.int64) for name in int64_features
    })
    features_dict.update({
        name: tf.VarLenFeature(tf.float32) for name in float32_features
    })
    features_dict.update({
        name: tf.FixedLenFeature([], tf.string) for name in bytes_features
    })
    features = tf.parse_single_example(
        serialized_example,
        features=features_dict)
    decoded_bytes = {}
    for name in bytes_features:
        decoded_bytes[name] = tf.decode_raw(features[name], tf.float32)
    obs_levels = tf.sparse_tensor_to_dense(features['obs_levels'])
    grid_3ds_shape = tf.cast(tf.sparse_tensor_to_dense(features['grid_3ds_shape']), tf.int32)
    grid_3ds_shape.set_shape((4,))
    grid_3ds = decoded_bytes['grid_3ds']
    grid_3ds = tf.reshape(grid_3ds, grid_3ds_shape)
    rewards = decoded_bytes['rewards']
    prob_rewards = decoded_bytes['prob_rewards']
    norm_rewards = decoded_bytes['norm_rewards']
    norm_prob_rewards = decoded_bytes['norm_prob_rewards']
    scores = decoded_bytes['scores']
    return obs_levels, grid_3ds, rewards, prob_rewards, norm_rewards, norm_prob_rewards, scores


def generate_tfrecords_from_batch(record_batch):
    for i in xrange(record_batch.grid_3ds.shape[0]):
        int64_mapping = {
            'obs_levels': record_batch.obs_levels,
            'grid_3ds_shape': record_batch.grid_3ds.shape[1:],
        }
        float32_mapping = {
        }
        bytes_mapping = {
            'grid_3ds': record_batch.grid_3ds[i, ...],
            'rewards': record_batch.rewards[i, ...],
            'prob_rewards': record_batch.prob_rewards[i, ...],
            'norm_rewards': record_batch.norm_rewards[i, ...],
            'norm_prob_rewards': record_batch.norm_prob_rewards[i, ...],
            'scores': record_batch.scores[i, ...],
        }
        feature_dict = {}
        feature_dict.update({
            name: _int64_array_feature(np.asarray(values).flatten()) for name, values in int64_mapping.iteritems()
        })
        feature_dict.update({
            name: _float32_array_feature(np.asarray(values).flatten()) for name, values in float32_mapping.iteritems()
        })
        feature_dict.update({
            name: _bytes_feature(array.tostring()) for name, array in bytes_mapping.iteritems()
        })
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        yield example