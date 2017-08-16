#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import threading
import Queue
import data_record
import file_helpers
import models
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


class TFDataProvider(object):

    def __init__(self, epoch_generator_factory,
                 batch_size,
                 grid_3d_shape,
                 num_epochs=-1):
        self._epoch_generator_factory = epoch_generator_factory
        self._batch_size = batch_size
        self._num_epochs = num_epochs

        self._grid_3d_in = tf.placeholder(dtype=tf.float32, shape=grid_3d_shape)
        self._action_in = tf.placeholder(dtype=tf.int32, shape=[])
        self._reward_in = tf.placeholder(dtype=tf.float32, shape=[])

        with tf.container("queue-container"):
            self._queue = tf.FIFOQueue(
                capacity=2000,
                dtypes=[tf.float32, tf.int32, tf.float32],
                shapes=[self._grid_3d_in.shape, self._action_in.shape, self._reward_in.shape])
            # self._queue = tf.RandomShuffleQueue(
            #     capacity=2000,
            #     min_after_dequeue=1000,
            #     dtypes=[tf.float32, tf.int32, tf.float32],
            #     shapes=[self._grid_3d_in.shape, self._action_in.shape, self._reward_in.shape])

        self._enqueue_op = self._queue.enqueue([self._grid_3d_in, self._action_in, self._reward_in])

        self._queue_batch_counts = []
        self._epoch = 0

        self._thread = None
        self._lock = threading.Lock()

    def _run(self, sess):
        """Thread loop. Feeds new data into queue."""
        def enqueue_epoch():
            epoch_generator = self._epoch_generator_factory()
            with self._lock:
                self._queue_batch_counts.append(0)
            for record_batch in epoch_generator:
                for j in xrange(record_batch.actions.shape[0]):
                    sess.run(self._enqueue_op, feed_dict={
                        self._action_in: record_batch.actions[j],
                        self._reward_in: record_batch.rewards[j],
                        self._grid_3d_in: record_batch.grid_3ds[j, ...]})
                with self._lock:
                    self._queue_batch_counts[-1] += 1
        if self._num_epochs <= 0:
            while True:
                enqueue_epoch()
        else:
            for i in xrange(self._num_epochs):
                enqueue_epoch()

    def start_thread(self, sess):
        """Start thread to fill queue """
        assert(self._thread is None)
        self._thread = threading.Thread(target=self._run, args=(sess,))
        self._thread.daemon = True  # Kill thread when all other threads stopped
        self._thread.start()
        return self._thread

    def join_thread(self):
        self._thread.join()
        self._thread = None

    def next_batch(self):
        return self._queue.dequeue_many(self._batch_size)

    def consume_batch(self):
        print(self._queue_batch_counts)
        print(self._epoch)
        with self._lock:
            assert(self._queue_batch_counts[0] > 0)
            self._queue_batch_counts[0] -= 1
            if self._queue_batch_counts[0] == 0:
                self._queue_batch_counts[0:1] = []
                self._epoch += 1
                return True
            return False
        # self._batch_count += 1
        # self._record_count += self._queue_batch_sizes[0]
        # self._queue_batch_sizes[0] = []

    def get_epoch(self):
        return self._epoch

    # def get_batch_count(self, sess):
    #     return self._batch_count
    #
    # def get_record_count(self, sess):
    #     return self._record_count


class DataProvider(object):

    def __init__(self, epoch_generator_factory,
                 grid_3d_shape,
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
        self._record_count += record_batch.actions.shape[0]
        return record_batch

    def get_queue_size(self):
        return self._batch_queue.qsize()

    def get_epoch(self):
        return self._epoch

    def get_batch_count(self):
        return self._batch_count

    def get_record_count(self):
        return self._record_count


def create_epoch_generator(filenames, grid_3d_mask, num_parallel_files=4,
                            shuffle_filenames=True, shuffle_records=True):
    if num_parallel_files < len(filenames):
        num_parallel_files = len(filenames)
    # Make a copy so we don't change the input list
    filenames = list(filenames)
    if shuffle_filenames:
        np.random.shuffle(filenames)
    for file_idx in xrange(0, len(filenames), num_parallel_files):
        current_filenames = filenames[file_idx:file_idx + num_parallel_files]
        for k in xrange(len(current_filenames)):
            records = data_record.read_hdf5_records(current_filenames[k])
            if k == 0:
                merged_grid_3ds = records.grid_3ds
                merged_actions = records.actions
                merged_rewards = records.rewards
            else:
                merged_grid_3ds = np.concatenate([merged_grid_3ds, records.grid_3ds])
                merged_actions = np.concatenate([merged_actions, records.actions])
                merged_rewards = np.concatenate([merged_rewards, records.rewards])
        merged_grid_3ds = merged_grid_3ds[..., grid_3d_mask]
        records = data_record.RecordBatch(records.obs_levels, merged_grid_3ds, merged_actions, merged_rewards)
        indices = np.arange(len(records.actions))
        if shuffle_records:
            np.random.shuffle(indices)
        for idx in xrange(len(indices)):
            action = records.actions[idx]
            reward = records.rewards[idx]
            grid_3d = records.grid_3ds[idx, ...]
            yield data_record.Record(records.obs_levels, grid_3d, action, reward)


def create_epoch_batch_generator(filenames, grid_3d_mask, batch_size, num_parallel_files=4,
                                 shuffle_filenames=True, shuffle_records=True):
    # filename_tf = tf.constant(filename, shape=[1])
    # filename_queue = tf.train.string_input_producer(filename_tf)
    # record = data_record.read_and_decode_tf_example(filename_queue)
    # records = data_record.read_hdf5_records_as_list(filename)
    if num_parallel_files > len(filenames):
        num_parallel_files = len(filenames)
    # num_parallel_files = 10
    # Make a copy so we don't change the input list
    filenames = list(filenames)
    if shuffle_filenames:
        np.random.shuffle(filenames)
    for file_idx in xrange(0, len(filenames), num_parallel_files):
        current_filenames = filenames[file_idx:file_idx + num_parallel_files]
        for k in xrange(num_parallel_files):
            records = data_record.read_hdf5_records(current_filenames[k])
            if k == 0:
                merged_grid_3ds = records.grid_3ds
                merged_actions = records.actions
                merged_rewards = records.rewards
            else:
                merged_grid_3ds = np.concatenate([merged_grid_3ds, records.grid_3ds])
                merged_actions = np.concatenate([merged_actions, records.actions])
                merged_rewards = np.concatenate([merged_rewards, records.rewards])
        merged_grid_3ds = merged_grid_3ds[..., grid_3d_mask]
        records = data_record.RecordBatch(records.obs_levels, merged_grid_3ds, merged_actions, merged_rewards)
        indices = np.arange(len(records.actions))
        if shuffle_records:
            np.random.shuffle(indices)
        for batch_idx in xrange(0, len(indices), batch_size):
            batch_actions = records.actions[batch_idx:batch_idx + batch_size]
            batch_rewards = records.rewards[batch_idx:batch_idx + batch_size]
            batch_grid_3ds = records.grid_3ds[batch_idx:batch_idx + batch_size, ...]
            yield data_record.RecordBatch(records.obs_levels, batch_grid_3ds, batch_actions, batch_rewards)


def run(args):
    num_epochs = 10000
    batch_size = 64
    validation_interval = 5
    save_interval = 50
    report_step_interval = 1
    # obs_levels_to_use = [0, 2, 4]
    obs_levels_to_use = [0, 1, 2, 4]
    grid_3d_mask = []
    for level in obs_levels_to_use:
        grid_3d_mask.append(2 * level)
        grid_3d_mask.append(2 * level + 1)

    train_path = "datasets/16x16x16_0-1-2-3-4-5/train"
    train_filename_generator = file_helpers.input_filename_generator_hdf5(train_path)
    train_filenames = list(train_filename_generator)
    if len(train_filenames) == 0:
        raise RuntimeError("No train dataset file")
    test_path = "datasets/16x16x16_0-1-2-3-4-5/test"
    test_filename_generator = file_helpers.input_filename_generator_hdf5(test_path)
    test_filenames = list(test_filename_generator)
    if len(test_filenames) == 0:
        raise RuntimeError("No test dataset file")

    record_batch = data_record.read_hdf5_records(train_filenames[0])
    grid_3d_batch = record_batch.grid_3ds[..., grid_3d_mask]
    grid_3d_shape = list(grid_3d_batch.shape[1:])

    global_step_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='global_step')
    inc_global_step = global_step_tf.assign_add(tf.constant(1, dtype=tf.int64))

    # epoch_generator_factory = lambda: create_epoch_generator(train_filenames, grid_3d_mask)
    epoch_generator_factory = lambda: create_epoch_batch_generator(train_filenames, grid_3d_mask, batch_size)
    with tf.device("/cpu:0"):
        data_provider = TFDataProvider(epoch_generator_factory, batch_size, grid_3d_shape)
    # data_provider.start_thread()

    test_epoch_generator_factory = lambda: create_epoch_batch_generator(test_filenames, grid_3d_mask, batch_size)
    test_data_provider = TFDataProvider(test_epoch_generator_factory, batch_size, grid_3d_shape, num_epochs=1)
    # test_data_provider.start_thread()

    # device_name = '/cpu:0'
    device_name = '/gpu:0'
    with tf.device(device_name):
        # grid_3d_tf = tf.placeholder(tf.float32, shape=[None] + list(grid_3d_shape), name="in_grid_3d")
        grid_3d_tf = data_provider.next_batch()[0]
        with tf.variable_scope("model"):
            conv3d_layer = models.Conv3DLayers(grid_3d_tf, dropout_rate=0.2)
            # conv3d_layer = models.Conv3DLayers(grid_3d_batch, dropout_rate=0.2)
            num_outputs = 6
            output_layer = models.RegressionOutputLayer(conv3d_layer.output, num_outputs)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # Generate ground-truth inputs for computing loss function
        # actions = tf.placeholder(dtype=np.int32, shape=[None], name="actions")
        # gt_rewards = tf.placeholder(dtype=np.float32, shape=[None], name="gt_rewards")
        actions = data_provider.next_batch()[1]
        gt_rewards = data_provider.next_batch()[2]
        actions_one_hot = tf.one_hot(actions, num_outputs, axis=-1, name="actions_one_hot")
        selected_output = tf.reduce_sum(tf.multiply(output_layer.output, actions_one_hot),
                                        axis=-1, name="selected_output")

        loss_batch = tf.square(selected_output - gt_rewards, name="loss_batch")
        loss = tf.reduce_mean(loss_batch, name="loss")
        loss_min = tf.reduce_min(loss_batch, name="loss_min")
        loss_max = tf.reduce_max(loss_batch, name="loss_max")

        # Generate output and loss function for testing (no dropout)
        with tf.variable_scope("model"):
            test_output_layer = models.RegressionOutputLayer(conv3d_layer.output_wo_dropout, num_outputs, reuse=True)
        test_selected_output = tf.reduce_sum(tf.multiply(test_output_layer.output, actions_one_hot),
                                        axis=-1, name="selected_output")
        test_loss_batch = tf.square(test_selected_output - gt_rewards, name="test_loss_batch")
        test_loss = tf.reduce_mean(test_loss_batch, name="test_loss")
        test_loss_min = tf.reduce_min(test_loss_batch, name="test_loss_min")
        test_loss_max = tf.reduce_max(test_loss_batch, name="test_loss_max")

    # Create optimizer
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss, var_list=variables)
    train_op = tf.group(train_op, inc_global_step)

    # Configure tensorflow
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    # config.log_device_placement = True

    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2.0)

    # Initialize tensorflow session
    run_metadata = tf.RunMetadata()
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    data_provider.start_thread(sess)
    test_data_provider.start_thread(sess)

    if args.restore:
        ckpt = tf.train.get_checkpoint_state(args.store_path)
        if ckpt is None:
            response = raw_input("WARNING: No previous checkpoint found. Continue? [y/n]")
            if response != "y":
                raise RuntimeError("Could not find previous checkpoint")
        else:
            print('Found previous checkpoint... restoring')
            saver.restore(sess, ckpt.model_checkpoint_path)

    timer = Timer()
    train_options = tf.RunOptions(timeout_in_ms=100000,
                                  trace_level=tf.RunOptions.FULL_TRACE)
    for epoch in xrange(num_epochs):
        compute_time = 0.0
        data_time = 0.0
        total_loss_value = 0.0
        total_loss_min = +np.finfo(np.float32).max
        total_loss_max = -np.finfo(np.float32).max
        assert(epoch == data_provider.get_epoch())
        # t1 = timer.elapsed_seconds()
        # record_batch = data_provider.next_batch()
        # t2 = timer.elapsed_seconds()
        # data_time += t2 - t1
        # while record_batch is not None:
        batch_count = 0
        epoch_done = False
        while not epoch_done:
            try:
                # _, loss_value = sess.run([train_op, loss], options=train_options)
                t1 = timer.elapsed_seconds()
                _, loss_v, loss_min_v, loss_max_v =\
                    sess.run([train_op, loss, loss_min, loss_max],
                             options=train_options, run_metadata=run_metadata, feed_dict={
                        # grid_3d_tf: record_batch.grid_3ds,
                        # actions: record_batch.actions,
                        # gt_rewards: record_batch.rewards,
                })
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                with open('timeline.ctf.json', 'w') as trace_file:
                    trace_file.write(trace.generate_chrome_trace_format())

                total_loss_value += loss_v
                total_loss_min = np.minimum(loss_min_v, total_loss_min)
                total_loss_max = np.maximum(loss_max_v, total_loss_max)
                t2 = timer.elapsed_seconds()
                compute_time += t2 - t1
                # record_count = data_provider.get_record_count()
                # batch_count = data_provider.get_batch_count()
                # if (record_count + 1) % report_step_interval == 0:
                #     print("batches: {}, record count: {}, data_time: {}, compute_time: {}".format(
                #         batch_count, record_count,
                #         data_time, compute_time))
                t1 = timer.elapsed_seconds()
                # record_batch = data_provider.next_batch()
                epoch_done = data_provider.consume_batch()
                t2 = timer.elapsed_seconds()
                data_time += t2 - t1
                batch_count += 1
            except tf.errors.DeadlineExceededError, exc:
                raise RuntimeError("Training timed out: {}".format(exc))
            # print("queue size:", data_provider.get_queue_size())
        # record_count = data_provider.get_record_count()
        # batch_count = data_provider.get_batch_count()
        total_loss_value /= batch_count
        print("train result:")
        print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
            epoch, total_loss_value, total_loss_min, total_loss_max))
        print("train timing:")
        print("  batches: {}, data time: {}, compute time: {}".format(
            batch_count, data_time, compute_time))

        # epoch_generator = create_epoch_generator(train_filenames, batch_size)
        # t1 = timer.elapsed_seconds()
        # record_count = 0
        # for batch_idx, record_batch in enumerate(epoch_generator):
        #     # grid_3d_batch = record_batch.grid_3ds[:, obs_levels_to_use, ...]
        #     grid_3d_batch = record_batch.grid_3ds[..., grid_3d_mask]
        #     # TODO: Gradient should be scaled to batch size, though this is a minor thing.
        #     # for record in records:
        #     #     print("  ", record.action, record.reward)
        #     t2 = timer.elapsed_seconds()
        #     data_time += t2 - t1
        #     _, loss_value = sess.run([train_op, loss], feed_dict={
        #         gt_rewards: record_batch.rewards,
        #         actions: record_batch.actions,
        #         grid_3d_tf: grid_3d_batch,
        #     })
        #     t1 = timer.elapsed_seconds()
        #     compute_time += t1 - t2
        #     record_count += grid_3d_batch.shape[0]
        #     if (batch_idx + 1) % report_step_interval == 0:
        #         print("batch: {}, record count: {}, data_time: {}, compute_time: {}".format(
        #             batch_idx, record_count, data_time, compute_time))

        if (epoch + 1) % validation_interval == 0:
            total_loss_value = 0.0
            total_loss_min = +np.finfo(np.float32).max
            total_loss_max = -np.finfo(np.float32).max
            # test_epoch_generator = create_epoch_batch_generator(train_filenames, grid_3d_mask, batch_size,
            #                                               shuffle_filenames=False, shuffle_records=False)
            # test_epoch_generator = create_epoch_batch_generator(test_filenames, grid_3d_mask, batch_size,
            #                                               shuffle_filenames=False, shuffle_records=False)
            # record_batch = test_data_provider.next_batch()
            # while record_batch is not None:
            batch_count = 0
            epoch_done = False
            while not epoch_done:
                # grid_3d_batch = record_batch.grid_3ds[:, obs_levels_to_use, ...]
                # grid_3d_batch = record_batch.grid_3ds[..., grid_3d_mask]
                loss_v, loss_min_v, loss_max_v = sess.run([test_loss, test_loss_min, test_loss_max], feed_dict={
                    grid_3d_tf: record_batch.grid_3ds,
                    actions: record_batch.actions,
                    gt_rewards: record_batch.rewards,
                })
                total_loss_value += loss_v
                total_loss_min = np.minimum(loss_min_v, total_loss_min)
                total_loss_max = np.maximum(loss_max_v, total_loss_max)
                # record_batch = test_data_provider.next_batch()
                epoch_done = data_provider.consume_batch()
                batch_count += 1
            # Restart test data provider
            test_data_provider.join_thread()
            test_data_provider.start_thread()
            # Compute average test loss
            # record_count = data_provider.get_record_count()
            # batch_count = data_provider.get_batch_count()
            total_loss_value /= batch_count
            print("------------")
            print("test result:")
            print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                epoch, total_loss_value, total_loss_min, total_loss_max))
            print("test timing:")
            print("batches: {}, data time: {}, compute time: {}".format(
                batch_count, data_time, compute_time))
            print("------------")

        if epoch > 0 and epoch % save_interval == 0:
            print("Saving model at epoch {}".format(epoch))
            saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)

    saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)
    data_provider.join_thread()


if __name__ == '__main__':
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--store-path', required=True, help='Store path')
    parser.add_argument('--restore', action="store_true", help='Restore previous model')

    args = parser.parse_args()

    run(args)
