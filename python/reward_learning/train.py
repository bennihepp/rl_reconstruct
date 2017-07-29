#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import time
import threading
import Queue
import data_record
import file_helpers
import models
import numpy as np
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging
import tensorflow.contrib.layers as tf_layers
import tf_utils
import data_provider
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


def create_epoch_batch_generator(filenames, batch_size,
                                 num_parallel_files=4,
                                 shuffle_filenames=True, shuffle_records=True,
                                 verbose=False):
    if num_parallel_files > len(filenames):
        num_parallel_files = len(filenames)
    # Make a copy so we don't change the input list
    filenames = list(filenames)
    if shuffle_filenames:
        np.random.shuffle(filenames)
    transfer_rate_report_interval = 10
    batch_count = 0
    t0 = time.time()
    for file_idx in xrange(0, len(filenames), num_parallel_files):
        current_filenames = filenames[file_idx:file_idx + num_parallel_files]
        for k in xrange(len(current_filenames)):
            records = data_record.read_hdf5_records_v2(current_filenames[k])
            if k == 0:
                merged_grid_3ds = records.grid_3ds
                merged_rewards = records.rewards
                merged_prob_rewards = records.prob_rewards
                merged_scores = records.scores
            else:
                merged_grid_3ds = np.concatenate([merged_grid_3ds, records.grid_3ds])
                merged_rewards = np.concatenate([merged_rewards, records.rewards])
                merged_prob_rewards = np.concatenate([merged_prob_rewards, records.prob_rewards])
                merged_scores = np.concatenate([merged_scores, records.scores])
        records = data_record.RecordBatch(
            records.obs_levels, merged_grid_3ds, merged_rewards, merged_prob_rewards, merged_scores)
        indices = np.arange(records.rewards.shape[0])
        if shuffle_records:
            np.random.shuffle(indices)
        for batch_idx in xrange(0, len(indices), batch_size):
            batch_grid_3ds = records.grid_3ds[batch_idx:batch_idx + batch_size, ...]
            batch_rewards = records.rewards[batch_idx:batch_idx + batch_size, ...]
            batch_prob_rewards = records.prob_rewards[batch_idx:batch_idx + batch_size, ...]
            batch_scores = records.scores[batch_idx:batch_idx + batch_size, ...]
            if batch_idx + batch_size > records.rewards.shape[0]:
                # Incomplete batch, repeat data
                while batch_rewards.shape[0] < batch_size:
                    # print("Padding batch to get desired batch size")
                    batch_rewards = np.concatenate([batch_rewards, batch_rewards])
                    batch_prob_rewards = np.concatenate([batch_prob_rewards, batch_prob_rewards])
                    batch_scores = np.concatenate([batch_scores, batch_scores])
                    batch_grid_3ds = np.concatenate([batch_grid_3ds, batch_grid_3ds])
                batch_rewards = batch_rewards[:batch_size, ...]
                batch_prob_rewards = batch_prob_rewards[:batch_size, ...]
                batch_scores = batch_scores[:batch_size, ...]
                batch_grid_3ds = batch_grid_3ds[:batch_size, ...]
            assert(batch_rewards.shape[0] == batch_size)
            assert(batch_prob_rewards.shape[0] == batch_size)
            assert(batch_scores.shape[0] == batch_size)
            assert(batch_grid_3ds.shape[0] == batch_size)
            if verbose:
                print("Queue filler batch_count:", batch_count)
            yield data_record.RecordBatch(records.obs_levels, batch_grid_3ds,
                                          batch_rewards, batch_prob_rewards,
                                          batch_scores)
            batch_count += 1
            if verbose:
                if batch_count % transfer_rate_report_interval == 0:
                    print("merged_grid_3ds.shape:", merged_grid_3ds.shape)
                    batch_bytes = 4 * len(batch_grid_3ds.flatten()) + 4 * len(batch_rewards.flatten()) \
                                 + 4 * len(batch_prob_rewards.flatten()) + 4 * len(batch_scores.flatten())
                    batch_per_sec = float(batch_count) / (time.time() - t0)
                    mb_per_sec = float(batch_bytes * batch_count) / (time.time() - t0) / 1024. / 1024.
                    print("Memory per batch: {} kB".format(batch_bytes / 1024.))
                    print("Read {} batches".format(batch_count))
                    print("Reading rate: {} batch/s, {} MB/s".format(batch_per_sec, mb_per_sec))
                    t0 = time.time()
                    batch_count = 0


def run(args):
    # Learning parameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_grad_global_norm = args.max_grad_global_norm
    validation_interval = args.validation_interval
    train_summary_interval = args.train_summary_interval
    model_summary_interval = args.model_summary_interval
    checkpoint_interval = args.checkpoint_interval

    create_tf_timeline = False

    split_data = args.test_data_path is None
    if split_data:
        train_test_split_ratio = args.train_test_split_ratio
        train_percentage = train_test_split_ratio / (1 + train_test_split_ratio)
        print("Train percentage: {}".format(train_percentage))
        filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
        filenames = sorted(list(filename_generator))
        if args.train_test_split_shuffle:
            np.random.shuffle(filenames)
        train_filenames = filenames[:int(len(filenames) * train_percentage)]
        test_filenames = filenames[int(len(filenames) * train_percentage):]
        all_filenames = filenames
    else:
        train_filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
        train_filenames = list(train_filename_generator)
        test_filename_generator = file_helpers.input_filename_generator_hdf5(args.test_data_path)
        test_filenames = list(test_filename_generator)
        all_filenames = train_filenames + test_filenames

    if len(train_filenames) == 0:
        raise RuntimeError("No train dataset file")
    else:
        print("Found {} train dataset files".format(len(train_filenames)))
    if len(test_filenames) == 0:
        raise RuntimeError("No test dataset file")
    else:
        print("Found {} test dataset files".format(len(test_filenames)))

    # Limit dataset size?
    if args.max_num_train_files > 0:
        train_filenames = train_filenames[:args.max_num_train_files]
        print("Using {} train dataset files".format(len(train_filenames)))
    else:
        print("Using all train dataset files")
    if args.max_num_test_files > 0:
        test_filenames = test_filenames[:args.max_num_test_files]
        print("Using {} test dataset files".format(len(test_filenames)))
    else:
        print("Using all test dataset files")

    # Input configuration
    grid_3d_channels = [0, 1]
    subvolume_slice_x = slice(0, 16)
    subvolume_slice_y = slice(0, 16)
    subvolume_slice_z = slice(0, 16)
    tmp_record_batch = data_record.read_hdf5_records_v2(train_filenames[0])
    print("obs_levels in data record: {}".format(tmp_record_batch.obs_levels))
    raw_grid_3d_batch = tmp_record_batch.grid_3ds
    if args.obs_levels_to_use is None:
        grid_3d_channels = range(raw_grid_3d_batch.shape[-1])
    else:
        obs_levels_to_use = [int(x) for x in args.obs_levels_to_use.split(',')]
        grid_3d_channels = []
        for level in obs_levels_to_use:
            grid_3d_channels.append(2 * level)
            grid_3d_channels.append(2 * level + 1)

    if args.subvolume_slice_x is None:
        subvolume_slice_x = slice(0, raw_grid_3d_batch.shape[1])
    else:
        subvolume_slice_x = slice(*[int(x) for x in args.subvolume_slice_x.split(',')])
    if args.subvolume_slice_y is None:
        subvolume_slice_y = slice(0, raw_grid_3d_batch.shape[2])
    else:
        subvolume_slice_y = slice(*[int(x) for x in args.subvolume_slice_y.split(',')])
    if args.subvolume_slice_z is None:
        subvolume_slice_z = slice(0, raw_grid_3d_batch.shape[3])
    else:
        subvolume_slice_z = slice(*[int(x) for x in args.subvolume_slice_z.split(',')])

    print("grid_3d_channels: {}".format(grid_3d_channels))
    print("subvolume_slice_x: {}".format(subvolume_slice_x))
    print("subvolume_slice_y: {}".format(subvolume_slice_y))
    print("subvolume_slice_z: {}".format(subvolume_slice_z))
    # subvolume_slice_x_indices = range(raw_grid_3d_batch.shape[1])[subvolume_slice_x]
    # subvolume_slice_y_indices = range(raw_grid_3d_batch.shape[2])[subvolume_slice_y]
    # subvolume_slice_z_indices = range(raw_grid_3d_batch.shape[3])[subvolume_slice_z]
    subvolume_slices = [subvolume_slice_x, subvolume_slice_y, subvolume_slice_z]

    def get_input_from_record(record):
        return record.grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, grid_3d_channels]

    def get_input_from_grid_3ds(grid_3ds):
        return grid_3ds[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, grid_3d_channels]

    def get_input_batch(record_batch):
        # merged_grid_3ds = merged_grid_3ds[..., grid_3d_channels]
        # merged_grid_3ds = merged_grid_3ds[:, subvolume_slices[0], subvolume_slices[1], subvolume_slices[2], :]
        return record_batch.grid_3ds[:, subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, grid_3d_channels]

    # TODO
    # def get_input_from_record(record_tuple):
    #     obs_levels, grid_3ds, rewards, prob_rewards, norm_rewards, norm_prob_rewards, scores = record_tuple
    #     print(subvolume_slice_x)
    #     print(grid_3d_channels)
    #     # single_input = grid_3ds[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, grid_3d_channels]
    #     print(grid_3ds.shape)
    #     single_input = grid_3ds[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, ...]
    #     single_input = tf.transpose(single_input)
    #     single_input = tf.gather(single_input, tf.constant(grid_3d_channels))
    #     single_input = tf.transpose(single_input)
    #     print(single_input.shape)
    #     single_input.set_shape((16, 16, 16, len(grid_3d_channels)))
    #     return single_input

    # TODO
    # def get_target_from_record(record_tuple):
    #     obs_levels, grid_3ds, rewards, prob_rewards, norm_rewards, norm_prob_rewards, scores = record_tuple
    #     if args.target_id == "rewards":
    #         return rewards
    #     elif args.target_id == "norm_rewards":
    #         return norm_rewards
    #     elif args.target_id == "prob_rewards":
    #         return prob_rewards
    #     elif args.target_id == "norm_prob_rewards":
    #         return norm_prob_rewards
    #     elif args.target_id == "score":
    #         return scores[0:1]
    #     elif args.target_id == "norm_score":
    #         return scores[1:2]
    #     elif args.target_id == "prob_score":
    #         return scores[2:3]
    #     elif args.target_id == "norm_prob_score":
    #         return scores[3:4]
    #     elif args.target_id == "mean_occupancy":
    #         flat_batch = grid_3ds[..., 0::2].reshape((grid_3ds.shape[0], -1))
    #         mean_occupancy = tf.reduce_mean(flat_batch)
    #         return mean_occupancy
    #     elif args.target_id == "sum_occupancy":
    #         flat_batch = grid_3ds[..., 0::2].reshape((grid_3ds.shape[0], -1))
    #         sum_occupancy = tf.reduce_sum(flat_batch)
    #         return sum_occupancy
    #     elif args.target_id == "mean_observation":
    #         gather_idx = tf.concat([tf.range(0, tf.shape(grid_3ds)[0]),
    #                              tf.range(0, tf.shape(grid_3ds)[1]),
    #                              tf.range(0, tf.shape(grid_3ds)[2]),
    #                              tf.range(1, 2, 2)], axis=0)
    #         values = tf.gather_nd(grid_3ds, gather_idx)
    #         mean_observation = tf.reduce_sum(values)
    #         return mean_observation
    #     elif args.target_id == "sum_observation":
    #         flat_batch = grid_3ds[..., 1::2].reshape((grid_3ds.shape[0], -1))
    #         sum_observation = tf.reduce_sum(flat_batch)
    #         return sum_observation

    if args.target_id == "rewards":
        def get_target_from_record(record):
            return record.rewards
        # TODO: Remove these
        def get_target_batch(record_batch):
            return record_batch.rewards
    elif args.target_id == "norm_rewards":
        def get_target_from_record(record):
            return record.norm_rewards
        def get_target_batch(record_batch):
            return record_batch.norm_rewards
    elif args.target_id == "prob_rewards":
        def get_target_from_record(record):
            return record.prob_rewards
        def get_target_batch(record_batch):
            return record_batch.prob_rewards
    elif args.target_id == "norm_prob_rewards":
        def get_target_from_record(record):
            return record.norm_prob_rewards
        def get_target_batch(record_batch):
            return record_batch.norm_prob_rewards
    elif args.target_id == "score":
        def get_target_from_record(record):
            return record.scores[0:1]
        def get_target_batch(record_batch):
            return record_batch.scores[:, 0:1]
    elif args.target_id == "norm_score":
        def get_target_from_record(record):
            return record.scores[1:2]
        def get_target_batch(record_batch):
            return record_batch.scores[:, 1:2]
    elif args.target_id == "prob_score":
        def get_target_from_record(record):
            return record.scores[2:3]
        def get_target_batch(record_batch):
            return record_batch.scores[:, 2:3]
    elif args.target_id == "norm_prob_score":
        def get_target_from_record(record):
            return record.scores[3:4]
        def get_target_batch(record_batch):
            return record_batch.scores[:, 3:4]
    elif args.target_id == "mean_occupancy":
        def get_target_from_record(record):
            return np.mean(record.grid_3d[..., 0::2]).reshape((1,))
        def get_target_batch(record_batch):
            flat_batch = record_batch.grid_3ds[..., 0::2].reshape((record_batch.grid_3ds.shape[0], -1))
            mean_occupancy = flat_batch.mean(axis=-1)
            return mean_occupancy[:, np.newaxis]
    elif args.target_id == "sum_occupancy":
        def get_target_from_record(record):
            return np.sum(record.grid_3d[..., 0::2]).reshape((1,))
        def get_target_batch(record_batch):
            flat_batch = record_batch.grid_3ds[..., 0::2].reshape((record_batch.grid_3ds.shape[0], -1))
            sum_occupancy = flat_batch.sum(axis=-1)
            return sum_occupancy[:, np.newaxis]
    elif args.target_id == "mean_observation":
        def get_target_from_record(record):
            return np.mean(record.grid_3d[..., 1::2]).reshape((1,))
        def get_target_batch(record_batch):
            flat_batch = record_batch.grid_3ds[..., 1::2].reshape((record_batch.grid_3ds.shape[0], -1))
            mean_observation = flat_batch.mean(axis=-1)
            return mean_observation[:, np.newaxis]
    elif args.target_id == "sum_observation":
        def get_target_from_record(record):
            return np.sum(record.grid_3d[..., 1::2]).reshape((1,))
        def get_target_batch(record_batch):
            flat_batch = record_batch.grid_3ds[..., 1::2].reshape((record_batch.grid_3ds.shape[0], -1))
            sum_observation = flat_batch.sum(axis=-1)
            return sum_observation[:, np.newaxis]

    grid_3d_batch = tmp_record_batch.grid_3ds
    grid_3d_shape = list(grid_3d_batch.shape[1:])
    input_shape = list(get_input_batch(tmp_record_batch).shape[1:])
    target_shape = list(get_target_batch(tmp_record_batch).shape[1:])
    print("grid_3d_shape: {}".format(grid_3d_shape))
    print("input_shape: {}".format(input_shape))
    print("target_shape: {}".format(target_shape))

    # Mean and stddev of input for normalization
    input_stats_filename = args.input_stats_filename
    if args.input_stats_filename is None:
        input_stats_filename = os.path.join(args.data_path, file_helpers.DEFAULT_HDF5_STATS_FILENAME)
    if os.path.isfile(input_stats_filename):
        numpy_dict = data_record.read_hdf5_file_to_numpy_dict(input_stats_filename)
        all_data_size = int(numpy_dict["all_data_size"])
        mean_grid_3d_np = numpy_dict["mean_grid_3d"]
        stddev_grid_3d_np = numpy_dict["stddev_grid_3d"]
        tmp_record_batch = data_record.read_hdf5_records_v2(all_filenames[0])
        assert(np.all(numpy_dict["mean_grid_3d"].shape == tmp_record_batch.grid_3ds.shape[1:]))
        print("all_data_size:", all_data_size)
    else:
        print("Computing data statistics")
        all_data_size = 0
        sum_grid_3d_np = np.zeros(grid_3d_shape)
        sq_sum_grid_3d_np = np.zeros(grid_3d_shape)
        for filename in all_filenames:
            tmp_record_batch = data_record.read_hdf5_records_v2(filename)
            grid_3ds = tmp_record_batch.grid_3ds
            sum_grid_3d_np += np.sum(grid_3ds, axis=0)
            sq_sum_grid_3d_np += np.sum(np.square(grid_3ds), axis=0)
            all_data_size += grid_3ds.shape[0]
        mean_grid_3d_np = sum_grid_3d_np / all_data_size
        stddev_grid_3d_np = (sq_sum_grid_3d_np - np.square(sum_grid_3d_np) / all_data_size) / (all_data_size - 1)
        stddev_grid_3d_np[np.abs(stddev_grid_3d_np) < 1e-5] = 1
        stddev_grid_3d_np = np.sqrt(stddev_grid_3d_np)
        print("all_data_size:", all_data_size)
        print("mean_grid_3d_np:", np.mean(mean_grid_3d_np.flatten()))
        print("stddev_grid_3d_np:", np.mean(stddev_grid_3d_np.flatten()))
        sum_z_score = np.zeros(grid_3d_shape)
        sq_sum_z_score = np.zeros(grid_3d_shape)
        for filename in all_filenames:
            tmp_record_batch = data_record.read_hdf5_records_v2(filename)
            grid_3ds = tmp_record_batch.grid_3ds
            z_score = (grid_3ds - mean_grid_3d_np[np.newaxis, ...]) / stddev_grid_3d_np[np.newaxis, ...]
            assert(np.all(np.isfinite(z_score.flatten())))
            sum_z_score += np.sum(z_score, axis=0)
            sq_sum_z_score += np.sum(np.square(z_score), axis=0)
        mean_z_score = sum_z_score / all_data_size
        stddev_z_score = (sq_sum_z_score - np.square(sum_z_score) / all_data_size) / (all_data_size - 1)
        stddev_z_score[np.abs(stddev_z_score) < 1e-5] = 1
        stddev_z_score = np.sqrt(stddev_z_score)
        if not np.all(np.abs(mean_z_score) < 1e-2):
            print("mean_z_score")
            print(mean_z_score)
            print(mean_z_score[np.abs(mean_z_score - 1) >= 1e-2])
            print(np.sum(np.abs(mean_z_score - 1) >= 1e-2))
        assert(np.all(np.abs(mean_z_score) < 1e-3))
        if not np.all(np.abs(stddev_z_score - 1) < 1e-2):
            print("stddev_z_score")
            print(stddev_z_score)
            print(stddev_z_score[np.abs(stddev_z_score - 1) >= 1e-2])
            print(stddev_z_score[np.abs(stddev_z_score - 1) >= 1e-2] < 1e-2)
            print(np.sum(np.abs(stddev_z_score - 1) >= 1e-2))
            print(np.max(np.abs(stddev_z_score - 1)))
        assert(np.all(np.abs(stddev_z_score - 1) < 1e-2))
        data_record.write_hdf5_file(input_stats_filename, {
            "all_data_size": np.array(all_data_size),
            "mean_grid_3d": mean_grid_3d_np,
            "stddev_grid_3d": stddev_grid_3d_np,
            "mean_z_score": mean_z_score,
            "stddev_z_score": stddev_z_score,
        })

    del tmp_record_batch

    mean_record = data_record.RecordV2(None, mean_grid_3d_np, None, None, None, None, None)
    mean_input_np = get_input_from_record(mean_record)
    stddev_record = data_record.RecordV2(None, stddev_grid_3d_np, None, None, None, None, None)
    stddev_input_np = get_input_from_record(stddev_record)

    # Setup TF step and epoch counters
    global_step_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='global_step')
    inc_global_step = global_step_tf.assign_add(tf.constant(1, dtype=tf.int64))
    epoch_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='epoch')
    inc_epoch = epoch_tf.assign_add(tf.constant(1, dtype=tf.int64))

    def wrap_generator(generator):
        for record_batch in generator:
            yield get_input_batch(record_batch), get_target_batch(record_batch)

    def train_epoch_generator_factory():
        generator = create_epoch_batch_generator(train_filenames, batch_size, verbose=args.verbose)
        return wrap_generator(generator)

    def test_epoch_generator_factory():
        generator = create_epoch_batch_generator(test_filenames, batch_size,
                                                 shuffle_filenames=False, shuffle_records=False,
                                                 verbose=args.verbose)
        return wrap_generator(generator)

    # TODO
    # def read_records_from_files(filenames, num_threads, shuffle=True, num_epochs=-1, verbose=False):
    #     if num_epochs <= 0:
    #         num_epochs = None
    #     num_files = len(filenames)
    #     filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=num_epochs)
    #     record_tuple = data_record.read_and_decode_tfrecords(filename_queue)
    #     return record_tuple

    # TODO
    # def preprocess_record(record_tuple):
    #     single_input = get_input_from_record(record_tuple)
    #     single_target = get_target_from_record(record_tuple)
    #     return single_input, single_target, tf.constant(0, dtype=tf.int64)

    def make_batches(tensors,
                     shapes,
                     batch_size,
                     num_threads, queue_capacity, min_after_dequeue=2000,
                     shuffle=True, verbose=False):
        if shuffle:
            batch_fn = tf.train.shuffle_batch
        else:
            batch_fn = tf.train.batch
        batch_tensors = batch_fn(tensors,
                                 batch_size,
                                 queue_capacity,
                                 min_after_dequeue,
                                 num_threads,
                                 shapes=shapes)
        return batch_tensors

    def gpu_preload_pipeline(input_batch, target_batch, gpu_device_name="/gpu:0"):
        with tf.device(gpu_device_name):
            gpu_staging_area = tf_staging.StagingArea(
                dtypes=[input_batch.dtype, target_batch.dtype],
                shapes=[input_batch.shape, target_batch.shape])
        gpu_preload_op = gpu_staging_area.put([input_batch, target_batch])
        gpu_input_batch, gpu_target_batch = gpu_staging_area.get()
        return gpu_preload_op, gpu_input_batch, gpu_target_batch

    # Configure tensorflow
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    config.intra_op_parallelism_threads = args.intra_op_parallelism
    config.inter_op_parallelism_threads = args.inter_op_parallelism
    # config.log_device_placement = True

    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()

    def preprocess_record(record, epoch):
        single_input = get_input_from_record(record)
        single_input = (single_input - mean_input_np) / stddev_input_np
        single_target = get_target_from_record(record)
        return single_input, single_target, epoch

    with tf.device("/cpu:0"):
        filename_queue_provider = tf_utils.FilenameQueueProvider(filenames, coord, shuffle=True, verbose=args.verbose)
        epoch_shape = []
        train_data_bridge = data_provider.TFDataBridge(
            sess, batch_size,
            [input_shape, target_shape, epoch_shape],
            [tf.float32, tf.float32, tf.int64],
            queue_capacity=args.cpu_train_queue_capacity,
            min_after_dequeue=2000,
            shuffle=True,
            name="train_data_queue")
        test_data_bridge = data_provider.TFDataBridge(
            sess, batch_size,
            [input_shape, target_shape, epoch_shape],
            [tf.float32, tf.float32, tf.int64],
            queue_capacity=args.cpu_test_queue_capacity,
            min_after_dequeue=0,
            shuffle=False,
            name="test_data_queue")

        def enqueue_record_with_epoch_factory(data_bridge):
            def enqueue_record_with_epoch(record, epoch):
                single_input, single_target = preprocess_record(record)
                epoch_tf = tf.constant(0, tf.int64)
                data_bridge.enqueue([single_input, single_target, epoch_tf])
            return enqueue_record_with_epoch

        train_hdf5_reader = data_record.HDF5QueueReader(
            filename_queue_provider.get_next,
            enqueue_record_with_epoch_factory(train_data_bridge),
            coord,
            verbose=args.verbose)
        test_hdf5_reader = data_record.HDF5QueueReader(
            filename_queue_provider.get_next,
            enqueue_record_with_epoch_factory(test_data_bridge),
            coord,
            verbose=args.verbose)
        train_tensors = train_data_bridge.deque()
        test_tensors = test_data_bridge.deque()
        train_input_batch, train_target_batch, train_epoch_batch = make_batches(
            train_tensors,
            [input_shape, target_shape, epoch_shape],
            batch_size,
            num_threads=2,
            queue_capacity=args.cpu_train_queue_capacity,
            verbose=args.verbose)
        test_input_batch, test_target_batch, test_epoch_batch = make_batches(
            test_tensors,
            [input_shape, target_shape, epoch_shape],
            batch_size,
            num_threads=2,
            queue_capacity=args.cpu_test_queue_capacity,
            verbose=args.verbose)
        train_max_epoch_batch = tf.reduce_max(train_epoch_batch)
        test_max_epoch_batch = tf.reduce_max(test_epoch_batch)
        train_gpu_preload_op, train_input_batch, train_target_batch = \
            gpu_preload_pipeline(train_input_batch, train_target_batch)
        test_gpu_preload_op, test_input_batch, test_target_batch = \
            gpu_preload_pipeline(test_input_batch, test_target_batch)

    # Create model

    # Parameters
    activation_fn_3dconv = tf_utils.get_activation_function_by_name(args.activation_fn_3dconv, tf.nn.relu)
    num_units_regression = [int(x) for x in args.num_units_regression.split(',')]
    activation_fn_regression = tf_utils.get_activation_function_by_name(args.activation_fn_regression, tf.nn.relu)

    if args.gpu_id < 0:
        device_name = '/cpu:0'
    else:
        device_name = '/gpu:{}'.format(args.gpu_id)
    with tf.device(device_name):
        # input_batch = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name="in_input")
        with tf.variable_scope("model"):
            with tf.variable_scope("conv3d"):
                conv3d_layer = models.Conv3DLayers(input_batch,
                                                   num_convs_per_block=args.num_convs_per_block,
                                                   initial_num_filters=args.initial_num_filters,
                                                   filter_increase_per_block=args.filter_increase_per_block,
                                                   filter_increase_within_block=args.filter_increase_within_block,
                                                   maxpool_after_each_block=args.maxpool_after_each_block,
                                                   max_num_blocks=args.max_num_blocks,
                                                   max_output_grid_size=args.max_output_grid_size,
                                                   add_biases=args.add_biases_3dconv,
                                                   activation_fn=activation_fn_3dconv,
                                                   dropout_rate=args.dropout_rate)
            num_outputs = target_shape[-1]
            with tf.variable_scope("regression"):
                output_layer = models.RegressionOutputLayer(conv3d_layer.output,
                                                            num_outputs,
                                                            num_units=num_units_regression,
                                                            activation_fn=activation_fn_regression)
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

        # Generate ground-truth inputs for computing loss function
        # target_batch = tf.placeholder(dtype=np.float32, shape=[None], name="target_batch")

        loss_batch = tf.reduce_mean(tf.square(output_layer.output - target_batch), axis=-1, name="loss_batch")
        loss = tf.reduce_mean(loss_batch, name="loss")
        loss_min = tf.reduce_min(loss_batch, name="loss_min")
        loss_max = tf.reduce_max(loss_batch, name="loss_max")

        # Generate output and loss function for testing (no dropout)
        with tf.variable_scope("model", reuse=True):
            with tf.variable_scope("conv3d"):
                test_conv3d_layer = models.Conv3DLayers(test_input_batch,
                                                        num_convs_per_block=args.num_convs_per_block,
                                                        initial_num_filters=args.initial_num_filters,
                                                        filter_increase_per_block=args.filter_increase_per_block,
                                                        filter_increase_within_block=args.filter_increase_within_block,
                                                        maxpool_after_each_block=args.maxpool_after_each_block,
                                                        max_num_blocks=args.max_num_blocks,
                                                        max_output_grid_size=args.max_output_grid_size,
                                                        add_biases=args.add_biases_3dconv,
                                                        activation_fn=activation_fn_3dconv,
                                                        dropout_rate=0.0)
            num_outputs = target_shape[-1]
            with tf.variable_scope("regression"):
                test_output_layer = models.RegressionOutputLayer(test_conv3d_layer.output_wo_dropout,
                                                                 num_outputs,
                                                                 num_units=num_units_regression,
                                                                 activation_fn=activation_fn_regression)
        test_loss_batch = tf.reduce_mean(tf.square(test_output_layer.output - test_target_batch),
                                         axis=-1, name="loss_batch")
        test_loss = tf.reduce_mean(test_loss_batch, name="test_loss")
        test_loss_min = tf.reduce_min(test_loss_batch, name="test_loss_min")
        test_loss_max = tf.reduce_max(test_loss_batch, name="test_loss_max")

        gradients = tf.gradients(loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_global_norm)
        # Create optimizer
        optimizer_class = tf_utils.get_optimizer_by_name(args.optimizer, tf.train.AdamOptimizer)
        learning_rate_tf = tf.train.exponential_decay(args.initial_learning_rate,
                                                      epoch_tf,
                                                      args.learning_rate_decay_epochs,
                                                      args.learning_rate_decay_rate,
                                                      args.learning_rate_decay_staircase)
        opt = optimizer_class(learning_rate_tf)
        gradients_and_variables = list(zip(gradients, variables))
        train_op = opt.apply_gradients(gradients_and_variables)
        # train_op = opt.minimize(loss, var_list=variables)
        train_op = tf.group(train_op, inc_global_step)

    # Tensorboard summaries
    summary_loss = tf.placeholder(tf.float32, [])
    summary_loss_min = tf.placeholder(tf.float32, [])
    summary_loss_max = tf.placeholder(tf.float32, [])
    summary_grad_global_norm = tf.placeholder(tf.float32, [])
    summary_var_global_norm = tf.placeholder(tf.float32, [])
    var_global_norm = tf.global_norm(variables)
    grad_global_norm = tf.global_norm(gradients)
    with tf.name_scope('training'):
        train_summary_op = tf.summary.merge([
            tf.summary.scalar("epoch", epoch_tf),
            tf.summary.scalar("learning_rate", learning_rate_tf),
            tf.summary.scalar("loss", summary_loss),
            tf.summary.scalar("loss_min", summary_loss_min),
            tf.summary.scalar("loss_max", summary_loss_max),
            tf.summary.scalar("grad_global_norm", summary_grad_global_norm),
            tf.summary.scalar("var_global_norm", summary_var_global_norm),
        ])
    with tf.name_scope('testing'):
        test_summary_op = tf.summary.merge([
            tf.summary.scalar("epoch", epoch_tf),
            tf.summary.scalar("loss", summary_loss),
            tf.summary.scalar("loss_min", summary_loss_min),
            tf.summary.scalar("loss_max", summary_loss_max),
        ])
    # Model histogram summaries
    # with tf.name_scope('model'):
    with tf.device("/cpu:0"):
        target_summary = tf.summary.histogram("target_batch", target_batch),
        model_summaries = [target_summary] + conv3d_layer.summaries + output_layer.summaries
        model_summary_op = tf.summary.merge(model_summaries)

    saver = tf.train.Saver(max_to_keep=args.keep_n_last_checkpoints,
                           keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours)

    # qr = tf.train.QueueRunner(queue, [enqueue_op] * 4)
    filename_queue_provider.start()
    hdf5_reader.start()
    custom_threads = [
        filename_queue_provider.thread,
        hdf5_reader.thread
    ]

    # Print all variables
    print("Tensorflow variables:")
    for var in tf.global_variables():
        print("  {}: {}".format(var.name, var.shape))

    # Initialize tensorflow session
    if create_tf_timeline:
        run_metadata = tf.RunMetadata()
    else:
        run_metadata = None
    init = tf.global_variables_initializer()
    sess.run(init)

    # enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
    enqueue_threads = []

    try:
        # Start data provider threads
        tf.train.start_queue_runners(sess=sess)
        # train_data_provider.start_thread(sess)
        # test_data_provider.start_thread(sess)

        # Tensorboard summary writer
        log_path = args.log_path if args.log_path is not None else args.store_path
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)

        if args.restore:
            # Try to restore model
            ckpt = tf.train.get_checkpoint_state(args.store_path)
            if ckpt is None:
                response = raw_input("WARNING: No previous checkpoint found. Continue? [y/n]")
                if response != "y":
                    raise RuntimeError("Could not find previous checkpoint")
            else:
                print('Found previous checkpoint... restoring')
                saver.restore(sess, ckpt.model_checkpoint_path)

        timer = Timer()
        # Create timelines for profiling?
        if create_tf_timeline:
            train_options = tf.RunOptions(timeout_in_ms=100000,
                                          trace_level=tf.RunOptions.FULL_TRACE)
        else:
            train_options = None

        mean_input, stddev_input = sess.run([mean_input_tf, stddev_input_tf])
        print("mean_input_tf:", np.mean(mean_input.flatten()))
        print("stddev_input_tf:", np.mean(stddev_input.flatten()))

        print("Preloading GPU")

        # Get preload ops and make sure the GPU pipeline is filled with a mini-batch
        train_op = tf.group(train_op, train_gpu_preload_op)
        sess.run([train_gpu_preload_op, test_gpu_preload_op])

        sess.graph.finalize()

        print("Starting training")

        initial_epoch = int(sess.run([epoch_tf])[0])
        for epoch in xrange(initial_epoch, num_epochs):
            if coord.should_stop():
                break

            compute_time = 0.0
            data_time = 0.0
            total_loss_value = 0.0
            total_loss_min = +np.finfo(np.float32).max
            total_loss_max = -np.finfo(np.float32).max
            batch_count = 0
            # assert(epoch == train_data_provider.get_epoch())
            assert(epoch == int(sess.run([train_max_epoch_batch])[0]))
            do_summary = epoch > 0 and epoch % train_summary_interval == 0
            if do_summary:
                var_global_norm_v = 0.0
                grad_global_norm_v = 0.0
            do_model_summary = epoch % model_summary_interval == 0
            # while True:
            epoch_done = False
            while not epoch_done:
                if args.verbose:
                    print("Training batch # {} in epoch {}. Record # {}".format(batch_count, epoch, batch_count * batch_size))
                train_fetches = [train_op, train_max_epoch_batch, loss, loss_min, loss_max]
                summary_fetches = []
                model_summary_fetches = []
                if do_summary:
                    summary_fetches = [var_global_norm, grad_global_norm]
                if do_model_summary and batch_count == 0:
                    model_summary_fetches = [model_summary_op]
                fetched =\
                    sess.run(train_fetches + summary_fetches + model_summary_fetches,
                             options=train_options, run_metadata=run_metadata)
                _, max_data_epoch, loss_v, loss_min_v, loss_max_v = fetched[:len(train_fetches)]
                fetch_offset = len(train_fetches)
                if do_summary:
                    var_global_norm_v += fetched[fetch_offset]
                    grad_global_norm_v += fetched[fetch_offset + 1]
                    fetch_offset += len(summary_fetches)
                if do_model_summary and batch_count == 0:
                    model_summary = fetched[fetch_offset]
                    # fetch_offset += len(model_summary_fetches)
                if create_tf_timeline:
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    with open('timeline.ctf.json', 'w') as trace_file:
                        trace_file.write(trace.generate_chrome_trace_format())

                total_loss_value += loss_v
                total_loss_min = np.minimum(loss_min_v, total_loss_min)
                total_loss_max = np.maximum(loss_max_v, total_loss_max)
                # epoch_done = train_data_provider.consume_batch()
                epoch_done = max_data_epoch > epoch
                if args.verbose:
                    print("Max data epoch: {}".format(max_data_epoch))
                batch_count += 1

            total_loss_value /= batch_count

            if do_summary:
                var_global_norm_v /= batch_count
                grad_global_norm_v /= batch_count
                summary, = sess.run([train_summary_op], feed_dict={
                    summary_loss: total_loss_value,
                    summary_loss_min: total_loss_min,
                    summary_loss_max: total_loss_max,
                    summary_var_global_norm: var_global_norm_v,
                    summary_grad_global_norm: grad_global_norm_v,
                })
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(summary, global_step=global_step)
                summary_writer.flush()
            if do_model_summary:
                print("Model summary")
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(model_summary, global_step=global_step)
                summary_writer.flush()

            print("train result:")
            print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                epoch, total_loss_value, total_loss_min, total_loss_max))
            print("train timing:")
            print("  batches: {}, data time: {}, compute time: {}".format(
                batch_count, data_time, compute_time))

            # Heartbeat signal for Philly cluster
            progress = float(epoch - initial_epoch) / (num_epochs - initial_epoch)
            print("PROGRESS: {:05.2f}%".format(progress))

            sess.run([inc_epoch])
            learning_rate = float(sess.run([learning_rate_tf])[0])
            print("Current learning rate: {:e}".format(learning_rate))

            if (epoch + 1) % validation_interval == 0:
                total_loss_value = 0.0
                total_loss_min = +np.finfo(np.float32).max
                total_loss_max = -np.finfo(np.float32).max
                batch_count = 0
                # while True:
                epoch_done = False
                while not epoch_done:
                    _, loss_v, loss_min_v, loss_max_v = sess.run([test_gpu_preload_op, test_loss, test_loss_min, test_loss_max])
                    total_loss_value += loss_v
                    total_loss_min = np.minimum(loss_min_v, total_loss_min)
                    total_loss_max = np.maximum(loss_max_v, total_loss_max)
                    epoch_done = test_max_epoch_batch > epoch
                    batch_count += 1
                total_loss_value /= batch_count
                print("------------")
                print("test result:")
                print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                    epoch, total_loss_value, total_loss_min, total_loss_max))
                print("test timing:")
                print("batches: {}, data time: {}, compute time: {}".format(
                    batch_count, data_time, compute_time))
                print("------------")
                summary, = sess.run([test_summary_op], feed_dict={
                    summary_loss: total_loss_value,
                    summary_loss_min: total_loss_min,
                    summary_loss_max: total_loss_max,
                })
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(summary, global_step=global_step)
                summary_writer.flush()

            if epoch > 0 and epoch % checkpoint_interval == 0:
                print("Saving model at epoch {}".format(epoch))
                saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)

        saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)

    except Exception, exc:
        print("Exception in training loop: {}".format(exc))
        coord.request_stop(exc)
    except KeyboardInterrupt:
        print("Keyboard interrupt in training loop")
    finally:
        print("Requesting stop")
        coord.request_stop()
        coord.join(enqueue_threads + custom_threads)


if __name__ == '__main__':
    np.set_printoptions(threshold=5)

    def argparse_bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU to be used (-1 for CPU).')
    parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--test-data-path', required=False, help='Test data path. If not provided data will be split.')
    parser.add_argument('--store-path', required=True, help='Store path.')
    parser.add_argument('--log-path', required=False, help='Log path.')
    parser.add_argument('--restore', action="store_true", help='Whether to restore existing model.')

    # Train test split
    parser.add_argument('--train_test_split_ratio', type=float, default=4.0)
    parser.add_argument('--train_test_split_shuffle', action="store_true")
    parser.add_argument('--max_num_train_files', type=int, default=-1)
    parser.add_argument('--max_num_test_files', type=int, default=-1)

    # Resource allocation
    parser.add_argument('--intra_op_parallelism', type=int, default=1)
    parser.add_argument('--inter_op_parallelism', type=int, default=4)
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.5)
    parser.add_argument('--cpu_train_queue_capacity', type=int, default=1024 * 128)
    parser.add_argument('--cpu_test_queue_capacity', type=int, default=1024 * 128)

    # Learning parameters
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--max_grad_global_norm', type=float, default=1e3)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=10)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.96)
    parser.add_argument('--learning_rate_decay_staircase', type=argparse_bool, default=False)

    # Report and checkpoint saving
    parser.add_argument('--validation_interval', type=int, default=10)
    parser.add_argument('--train_summary_interval', type=int, default=10)
    parser.add_argument('--model_summary_interval', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--keep_checkpoint_every_n_hours', type=float, default=2)
    parser.add_argument('--keep_n_last_checkpoints', type=int, default=5)

    # Data parameters
    parser.add_argument('--target_id', type=str, default="prob_rewards")
    parser.add_argument('--input_stats_filename', type=str)
    parser.add_argument('--obs_levels_to_use', type=str)
    parser.add_argument('--subvolume_slice_x', type=str)
    parser.add_argument('--subvolume_slice_y', type=str)
    parser.add_argument('--subvolume_slice_z', type=str)

    # Model parameters
    parser.add_argument('--num_convs_per_block', type=int, default=2)
    parser.add_argument('--initial_num_filters', type=int, default=8)
    parser.add_argument('--filter_increase_per_block', type=int, default=8)
    parser.add_argument('--filter_increase_within_block', type=int, default=0)
    parser.add_argument('--maxpool_after_each_block', type=argparse_bool, default=False)
    parser.add_argument('--max_num_blocks', type=int, default=-1)
    parser.add_argument('--max_output_grid_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--add_biases_3dconv', type=argparse_bool, default=False)
    parser.add_argument('--activation_fn_3dconv', type=str, default="relu")
    parser.add_argument('--num_units_regression', type=str, default="1024")
    parser.add_argument('--activation_fn_regression', type=str, default="relu")

    args = parser.parse_args()

    run(args)
