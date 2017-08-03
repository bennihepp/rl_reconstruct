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
import tensorflow.contrib.memory_stats as tf_memory_stats
import tf_utils
import data_provider
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


def run(args):
    retry_save_wait_time = 3
    max_num_summary_errors = 3
    model_summary_num_batches = args.model_summary_num_batches
    if model_summary_num_batches <= 0:
        model_summary_num_batches = np.iinfo(np.int32).max

    # Learning parameters
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    max_grad_global_norm = args.max_grad_global_norm
    validation_interval = args.validation_interval
    train_summary_interval = args.train_summary_interval
    model_summary_interval = args.model_summary_interval
    checkpoint_interval = args.checkpoint_interval

    # Only for debugging of the train operation
    create_tf_timeline = False

    # Determine train/test dataset filenames and optionally split them
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
        if args.data_path == args.test_data_path:
            test_filenames = list(train_filenames)
        else:
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

    # Input configuration, i.e. which slices and channels from the 3D grids to use.
    # First read any of the data records
    tmp_record_batch = data_record.read_hdf5_records_v2(train_filenames[0])
    assert(tmp_record_batch.grid_3ds.shape[0] > 0)
    tmp_records = list(data_record.generate_single_records_from_batch_v2(tmp_record_batch))
    tmp_record = tmp_records[0]
    print("obs_levels in data record: {}".format(tmp_record.obs_levels))
    raw_grid_3d = tmp_record.grid_3d
    # Determine subvolume slices
    if args.subvolume_slice_x is None:
        subvolume_slice_x = slice(0, raw_grid_3d.shape[0])
    else:
        subvolume_slice_x = slice(*[int(x) for x in args.subvolume_slice_x.split(',')])
    if args.subvolume_slice_y is None:
        subvolume_slice_y = slice(0, raw_grid_3d.shape[1])
    else:
        subvolume_slice_y = slice(*[int(x) for x in args.subvolume_slice_y.split(',')])
    if args.subvolume_slice_z is None:
        subvolume_slice_z = slice(0, raw_grid_3d.shape[2])
    else:
        subvolume_slice_z = slice(*[int(x) for x in args.subvolume_slice_z.split(',')])
    # Determine channels to use
    if args.obs_levels_to_use is None:
        grid_3d_channels = range(raw_grid_3d.shape[-1])
    else:
        obs_levels_to_use = [int(x) for x in args.obs_levels_to_use.split(',')]
        grid_3d_channels = []
        for level in obs_levels_to_use:
            grid_3d_channels.append(2 * level)
            grid_3d_channels.append(2 * level + 1)
    # Print used subvolume slices and channels
    print("subvolume_slice_x: {}".format(subvolume_slice_x))
    print("subvolume_slice_y: {}".format(subvolume_slice_y))
    print("subvolume_slice_z: {}".format(subvolume_slice_z))
    print("grid_3d_channels: {}".format(grid_3d_channels))
    print("target_id: {}".format(args.target_id))

    # Retrieval functions for input and target from data records

    def get_input_from_record(record):
        return record.grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, grid_3d_channels]

    if args.target_id == "rewards":
        def get_target_from_record(record):
            return record.rewards
    elif args.target_id == "norm_rewards":
        def get_target_from_record(record):
            return record.norm_rewards
    elif args.target_id == "prob_rewards":
        def get_target_from_record(record):
            return record.prob_rewards
    elif args.target_id == "norm_prob_rewards":
        def get_target_from_record(record):
            return record.norm_prob_rewards
    elif args.target_id == "score":
        def get_target_from_record(record):
            return record.scores[0:1]
    elif args.target_id == "norm_score":
        def get_target_from_record(record):
            return record.scores[1:2]
    elif args.target_id == "prob_score":
        def get_target_from_record(record):
            return record.scores[2:3]
    elif args.target_id == "norm_prob_score":
        def get_target_from_record(record):
            return record.scores[3:4]
    elif args.target_id == "mean_occupancy":
        def get_target_from_record(record):
            return np.mean(record.grid_3d[..., 0::2]).reshape((1,))
    elif args.target_id == "sum_occupancy":
        def get_target_from_record(record):
            return np.sum(record.grid_3d[..., 0::2]).reshape((1,))
    elif args.target_id == "mean_observation":
        def get_target_from_record(record):
            return np.mean(record.grid_3d[..., 1::2]).reshape((1,))
    elif args.target_id == "sum_observation":
        def get_target_from_record(record):
            return np.sum(record.grid_3d[..., 1::2]).reshape((1,))
    else:
        raise NotImplementedError("Unknown target name: {}".format(args.target_id))

    # Report some stats on input and outputs for the first data file
    # This is only for sanity checking
    for channel in grid_3d_channels:
        print("mean channel {}: {}".format(channel, np.mean(tmp_record_batch.grid_3ds[..., channel])))
        print("stddev channel {}: {}".format(channel, np.std(tmp_record_batch.grid_3ds[..., channel])))
        print("min channel {}: {}".format(channel, np.min(tmp_record_batch.grid_3ds[..., channel])))
        print("max channel {}: {}".format(channel, np.max(tmp_record_batch.grid_3ds[..., channel])))
    tmp_inputs = [get_input_from_record(record) for record in tmp_records]
    for i in xrange(tmp_inputs[0].shape[-1]):
        values = [input[..., i] for input in tmp_inputs]
        print("mean for input {}: {}".format(i, np.mean(values)))
        print("stddev for input {}: {}".format(i, np.std(values)))
        print("min for input {}: {}".format(i, np.min(values)))
        print("max for input {}: {}".format(i, np.max(values)))
    tmp_targets = [get_target_from_record(record) for record in tmp_records]
    for i in xrange(tmp_targets[0].shape[-1]):
        values = [target[..., i] for target in tmp_targets]
        print("mean for target {}: {}".format(i, np.mean(values)))
        print("stddev for target {}: {}".format(i, np.std(values)))
        print("min for target {}: {}".format(i, np.min(values)))
        print("max for target {}: {}".format(i, np.max(values)))

    # Retrieve input and target shapes
    grid_3d = tmp_record.grid_3d
    grid_3d_shape = list(grid_3d.shape)
    input_shape = list(get_input_from_record(tmp_record).shape)
    target_shape = list(get_target_from_record(tmp_record).shape)
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

    # Configure tensorflow
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    config.intra_op_parallelism_threads = args.intra_op_parallelism
    config.inter_op_parallelism_threads = args.inter_op_parallelism
    # config.log_device_placement = True

    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()

    class InputPipeline(object):

        def _dequeue_record_factory(self, hdf5_input_pipeline):
            def dequeue_record():
                record = hdf5_input_pipeline.get_next_record()
                return record

            return dequeue_record

        def _preprocess_record(self, record):
            single_input = get_input_from_record(record)
            single_input = (single_input - mean_input_np) / stddev_input_np
            single_target = get_target_from_record(record)
            # single_input = np.ones(single_input.shape)
            # single_target = -np.ones(single_target.shape)
            return single_input, single_target

        def _enqueue_record_with(self, data_bridge):
            def enqueue_record(record):
                single_input, single_target = self._preprocess_record(record)
                return data_bridge.enqueue([single_input, single_target])

            return enqueue_record

        def _gpu_preload_pipeline(self, input_batch, target_batch, gpu_device_name="/gpu:0"):
            with tf.device(gpu_device_name):
                gpu_staging_area = tf_staging.StagingArea(
                    dtypes=[input_batch.dtype, target_batch.dtype],
                    shapes=[input_batch.shape, target_batch.shape])
            gpu_preload_op = gpu_staging_area.put([input_batch, target_batch])
            gpu_input_batch, gpu_target_batch = gpu_staging_area.get()
            return gpu_preload_op, gpu_input_batch, gpu_target_batch

        def __init__(self, filenames, queue_capacity, min_after_dequeue, shuffle, num_threads, name):
            with tf.device("/cpu:0"):
                # Create HDF5 readers
                self._hdf5_input_pipeline = data_record.HDF5ReaderProcessCoordinator(
                    filenames, coord, shuffle=shuffle, timeout=args.async_timeout,
                    num_processes=num_threads, verbose=args.verbose)
                self._num_records = None

                # Create python-TF data bridge
                assert(queue_capacity > min_after_dequeue)
                self._data_bridge = data_provider.TFDataBridge(
                    sess, batch_size,
                    [input_shape, target_shape],
                    [tf.float32, tf.float32],
                    queue_capacity=queue_capacity,
                    min_after_dequeue=min_after_dequeue,
                    shuffle=shuffle,
                    timeout=args.async_timeout,
                    name="{}_data_queue".format(name),
                    verbose=args.verbose)

                self._queue_bridges = [tf_utils.QueueBridge(
                    self._dequeue_record_factory(self._hdf5_input_pipeline),
                    self._enqueue_record_with(self._data_bridge),
                    coord, args.verbose) for _ in xrange(num_threads)]

                # Retrieve tensors from data bridge
                self._tensors = self._data_bridge.deque_batch()
                self._input_batch, self._target_batch = self._tensors
                # Generate GPU preload operations
                self._gpu_preload_op, self._input_batch, self._target_batch = \
                    self._gpu_preload_pipeline(self._input_batch, self._target_batch)

        def start(self):
            self._hdf5_input_pipeline.start()
            for bridge in self._queue_bridges:
                bridge.start()

        @property
        def num_records(self):
            if self._num_records is None:
                self._num_records = self._hdf5_input_pipeline.compute_num_records()
            return self._num_records

        @property
        def gpu_preload_op(self):
            return self._gpu_preload_op

        @property
        def input_batch(self):
            return self._input_batch

        @property
        def target_batch(self):
            return self._target_batch

        @property
        def threads(self):
            return [bridge.thread for bridge in self._queue_bridges] \
                   + [self._hdf5_input_pipeline.thread]

    with tf.device("/cpu:0"):
        train_input_pipeline = InputPipeline(
            train_filenames, args.cpu_train_queue_capacity, args.cpu_train_queue_min_after_dequeue,
            shuffle=True, num_threads=args.cpu_train_queue_threads, name="train")
        test_input_pipeline = InputPipeline(
            test_filenames, args.cpu_test_queue_capacity, args.cpu_test_queue_min_after_dequeue,
            shuffle=True, num_threads=args.cpu_test_queue_threads, name="test")
        print("# records in train dataset: {}".format(train_input_pipeline.num_records))
        print("# records in test dataset: {}".format(test_input_pipeline.num_records))

    # Create model

    if args.gpu_id < 0:
        device_name = '/cpu:0'
    else:
        device_name = '/gpu:{}'.format(args.gpu_id)
    with tf.device(device_name):
        # input_batch = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name="in_input")
        with tf.variable_scope("model"):
            model_config = args
            train_model = models.Model(model_config, train_input_pipeline.input_batch,
                                       target_shape=target_shape, is_training=True)
            variables = train_model.variables

        # Generate ground-truth inputs for computing loss function
        # train_target_batch = tf.placeholder(dtype=np.float32, shape=[None], name="target_batch")

        loss_batch = tf.reduce_mean(
            tf.square(train_model.output - train_input_pipeline.target_batch),
            axis=-1, name="loss_batch")
        loss = tf.reduce_mean(loss_batch, name="loss")
        loss_min = tf.reduce_min(loss_batch, name="loss_min")
        loss_max = tf.reduce_max(loss_batch, name="loss_max")

        # Generate output and loss function for testing (no dropout)
        with tf.variable_scope("model", reuse=True):
            model_config = args
            test_model = models.Model(model_config, test_input_pipeline.input_batch,
                                      target_shape=target_shape, is_training=False)
        test_loss_batch = tf.reduce_mean(
            tf.square(test_model.output - test_input_pipeline.target_batch),
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
        var_to_grad_dict = {var: grad for grad, var in gradients_and_variables}
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(gradients_and_variables, global_step=global_step_tf)

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

    class ModelHistogramSummary(object):

        def __init__(self, name_tensor_dict):
            self.fetches = []
            self.placeholders = []
            summaries = []
            for name, tensor in name_tensor_dict.iteritems():
                var_batch_shape = [None] + tensor.shape[1:].as_list()
                placeholder = tf.placeholder(tensor.dtype, var_batch_shape)
                summaries.append(tf.summary.histogram(name, placeholder))
                self.fetches.append(tensor)
                self.placeholders.append(placeholder)
            self.summary_op = tf.summary.merge(summaries)

    if args.verbose:
        print("Tensorflow variables:")
        for var in tf.global_variables():
            print("  {}: {}".format(var.name, var.shape))
    print("Model variables:")
    for grad, var in gradients_and_variables:
        print("  {}: {}".format(var.name, var.shape))

    # Model histogram summaries
    if model_summary_interval > 0:
        with tf.device("/cpu:0"):
            def get_model_hist_summary(input_pipeline, model):
                model_summary_dict = {"target_batch": input_pipeline.target_batch}
                for i, channel in enumerate(grid_3d_channels):
                    model_summary_dict.update({"input_batch/[{}]".format(channel):
                                                   input_pipeline.input_batch[..., slice(i, i+1)]})
                model_summary_dict.update({"input_batch/mean": input_pipeline.input_batch})
                for name, tensor in model.summaries.iteritems():
                    model_summary_dict["model/" + name] = tensor
                    # Reuse existing gradient expressions
                    if tensor in var_to_grad_dict:
                        grad = var_to_grad_dict[tensor]
                        if grad is not None:
                            model_summary_dict["model/" + name + "_grad"] = grad
                model_hist_summary = ModelHistogramSummary(model_summary_dict)
                return model_hist_summary
            train_model_hist_summary = get_model_hist_summary(train_input_pipeline, train_model)

    saver = tf.train.Saver(max_to_keep=args.keep_n_last_checkpoints,
                           keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours)

    try:
        train_input_pipeline.start()
        test_input_pipeline.start()
        custom_threads = train_input_pipeline.threads + test_input_pipeline.threads

        # Print model size etc.
        model_size = 0
        model_grad_size = 0
        for grad, var in gradients_and_variables:
            model_size += np.sum(tf_utils.variable_size(var))
            if grad is not None:
                model_grad_size += np.sum(tf_utils.variable_size(grad))
        model_conv3d_size = np.sum([tf_utils.variable_size(var) for var in train_model.modules["conv3d"].variables])
        model_regression_size = np.sum([tf_utils.variable_size(var) for var in train_model.modules["regression"].variables])
        print("Model variables: {} ({} MB)".format(model_size, model_size / 1024. / 1024.))
        print("Model gradients: {} ({} MB)".format(model_grad_size, model_grad_size / 1024. / 1024.))
        print("Model conv3d variables: {} ({} MB)".format(model_conv3d_size, model_conv3d_size / 1024. / 1024.))
        print("Model regression variables: {} ({} MB)".format(model_regression_size, model_regression_size / 1024. / 1024.))

        # Initialize tensorflow session
        if create_tf_timeline:
            run_metadata = tf.RunMetadata()
        else:
            run_metadata = None
        init = tf.global_variables_initializer()
        sess.run(init)

        # Start data provider threads
        # enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        custom_threads.extend(tf.train.start_queue_runners(sess=sess))

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

        # Create timelines for profiling?
        if create_tf_timeline:
            train_options = tf.RunOptions(timeout_in_ms=100000,
                                          trace_level=tf.RunOptions.FULL_TRACE)
        else:
            train_options = None

        # Get preload ops and make sure the GPU pipeline is filled with a mini-batch
        train_op = tf.group(train_op, train_input_pipeline.gpu_preload_op)

        with tf.device(tf_utils.gpu_device_name()):
            max_memory_gpu = tf_memory_stats.BytesLimit()
            peak_use_memory_gpu = tf_memory_stats.MaxBytesInUse()

        sess.graph.finalize()

        print("Preloading GPU")
        sess.run([train_input_pipeline.gpu_preload_op, test_input_pipeline.gpu_preload_op])

        print("Starting training")

        timer = Timer()
        initial_epoch = int(sess.run([epoch_tf])[0])
        num_summary_errors = 0
        total_batch_count = 0
        total_record_count = 0
        # Training loop for all epochs
        for epoch in xrange(initial_epoch, num_epochs):
            if coord.should_stop():
                break

            total_loss_value = 0.0
            total_loss_min = +np.finfo(np.float32).max
            total_loss_max = -np.finfo(np.float32).max
            batch_count = 0

            do_summary = train_summary_interval > 0 and epoch % train_summary_interval == 0
            if do_summary:
                var_global_norm_v = 0.0
                grad_global_norm_v = 0.0
                if args.verbose:
                    print("Generating train summary")

            do_model_summary = model_summary_interval > 0 and epoch % model_summary_interval == 0
            if do_model_summary:
                model_summary_fetched = [[] for _ in train_model_hist_summary.fetches]
                if args.verbose:
                    print("Generating model summary")
            # do_summary = False
            # do_model_summary = False

            # Training loop for one epoch
            for record_count in xrange(0, train_input_pipeline.num_records, batch_size):
                if args.verbose:
                    print("Training batch # {}, epoch {}, record # {}".format(batch_count, epoch, record_count))
                    print("  Total batch # {}, total record # {}".format(total_batch_count, total_record_count))

                # Create train op list
                fetches = [train_op, loss, loss_min, loss_max]

                if do_summary:
                    summary_offset = len(fetches)
                    fetches.extend([var_global_norm, grad_global_norm])

                if do_model_summary and batch_count < model_summary_num_batches:
                    model_summary_offset = len(fetches)
                    fetches.extend(train_model_hist_summary.fetches)

                fetched =\
                    sess.run(fetches,
                             options=train_options,
                             run_metadata=run_metadata)
                _, loss_v, loss_min_v, loss_max_v = fetched[:4]

                if do_summary:
                    var_global_norm_v += fetched[summary_offset]
                    grad_global_norm_v += fetched[summary_offset + 1]

                if do_model_summary and batch_count < model_summary_num_batches:
                    for i, value in enumerate(fetched[model_summary_offset:]):
                        # Make sure we copy the model summary tensors (otherwise it might be pinned GPU memory)
                        model_summary_fetched[i].append(np.array(value))

                if create_tf_timeline:
                    # Save timeline traces
                    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    with open('timeline.ctf.json', 'w') as trace_file:
                        trace_file.write(trace.generate_chrome_trace_format())

                if args.verbose and batch_count == 0:
                    # Print memory usage
                    max_memory_gpu_v, peak_use_memory_gpu_v = sess.run([max_memory_gpu, peak_use_memory_gpu])
                    print("Max memory on GPU: {} MB, peak used memory on GPU: {} MB".format(
                        max_memory_gpu_v / 1024. / 1024., peak_use_memory_gpu_v / 1024. / 1024.))

                total_loss_value += loss_v
                total_loss_min = np.minimum(loss_min_v, total_loss_min)
                total_loss_max = np.maximum(loss_max_v, total_loss_max)
                batch_count += 1
                total_batch_count += 1

            total_record_count += train_input_pipeline.num_records
            total_loss_value /= batch_count

            if do_summary:
                if args.verbose:
                    print("Writing train summary")
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
                try:
                    summary_writer.add_summary(summary, global_step=global_step)
                    summary_writer.flush()
                    num_summary_errors = 0
                except Exception, exc:
                    print("ERROR: Exception when trying to write model summary: {}".format(exc))
                    if num_summary_errors >= max_num_summary_errors:
                        print("Too many summary errors occured. Aborting.")
                        raise
            if do_model_summary:
                feed_dict = {}
                print("Concatenating model summaries")
                for i, placeholder in enumerate(train_model_hist_summary.placeholders):
                    feed_dict[placeholder] = np.concatenate(model_summary_fetched[i], axis=0)
                if args.verbose:
                    print("Building model summary")
                model_summary = sess.run([train_model_hist_summary.summary_op], feed_dict=feed_dict)[0]
                if args.verbose:
                    print("Writing model summary")
                global_step = int(sess.run([global_step_tf])[0])
                try:
                    summary_writer.add_summary(model_summary, global_step=global_step)
                    summary_writer.flush()
                    num_summary_errors = 0
                except Exception, exc:
                    print("ERROR: Exception when trying to write model summary: {}".format(exc))
                    if num_summary_errors >= max_num_summary_errors:
                        print("Too many summary errors occured. Aborting.")
                        raise
                del model_summary_fetched
                del feed_dict
                del model_summary

            print("train result:")
            print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                epoch, total_loss_value, total_loss_min, total_loss_max))
            epoch_time = timer.restart() / 60.
            print("train stats:")
            print("  batches: {}, records: {}, time: {} min".format(
                batch_count, record_count, epoch_time))
            print("  total batches: {}, total records: {}".format(
                total_batch_count, total_record_count))

            # Heartbeat signal for Philly cluster
            progress = 100 * float(epoch - initial_epoch) / (num_epochs - initial_epoch)
            print("PROGRESS: {:05.2f}%".format(progress))

            sess.run([inc_epoch])
            learning_rate = float(sess.run([learning_rate_tf])[0])
            print("Current learning rate: {:e}".format(learning_rate))

            if (epoch + 1) % validation_interval == 0:
                total_loss_value = 0.0
                total_loss_min = +np.finfo(np.float32).max
                total_loss_max = -np.finfo(np.float32).max
                batch_count = 0
                for record_count in xrange(0, test_input_pipeline.num_records, batch_size):
                    _, loss_v, loss_min_v, loss_max_v = sess.run([
                        test_input_pipeline.gpu_preload_op,
                        test_loss, test_loss_min, test_loss_max])
                    total_loss_value += loss_v
                    total_loss_min = np.minimum(loss_min_v, total_loss_min)
                    total_loss_max = np.maximum(loss_max_v, total_loss_max)
                    batch_count += 1
                total_loss_value /= batch_count
                summary, = sess.run([test_summary_op], feed_dict={
                    summary_loss: total_loss_value,
                    summary_loss_min: total_loss_min,
                    summary_loss_max: total_loss_max,
                })
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(summary, global_step=global_step)
                summary_writer.flush()
                print("------------")
                print("test result:")
                print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                    epoch, total_loss_value, total_loss_min, total_loss_max))
                record_count = batch_count * batch_size
                epoch_time_min = timer.restart() / 60.
                print("test stats:")
                print("  batches: {}, records: {}, time: {} min".format(
                    batch_count, record_count, epoch_time_min))
                print("------------")

            if epoch > 0 and epoch % checkpoint_interval == 0:
                saved = False
                trials = 0
                while not saved and trials < args.max_checkpoint_save_trials:
                    try:
                        trials += 1
                        print("Saving model at epoch {}".format(epoch))
                        saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)
                        saved = True
                        save_time = timer.restart()
                        print("Saving took {} s".format(save_time))
                    except Exception, exc:
                        print("ERROR: Exception when trying to save model: {}".format(exc))
                        if trials < args.max_checkpoint_save_trials:
                            print("Retrying to save model in {} s...".format(retry_save_wait_time))
                            time.sleep(retry_save_wait_time)
                        else:
                            raise

        print("Saving final model")
        saver.save(sess, os.path.join(args.store_path, "model"), global_step=global_step_tf)

    except Exception, exc:
        print("Exception in training loop: {}".format(exc))
        coord.request_stop(exc)
        raise exc
    finally:
        print("Requesting stop")
        coord.request_stop()
        coord.join(custom_threads, stop_grace_period_secs=args.async_timeout)


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

    # Report and checkpoint saving
    parser.add_argument('--async_timeout', type=int, default=10 * 60)
    parser.add_argument('--validation_interval', type=int, default=10)
    parser.add_argument('--train_summary_interval', type=int, default=10)
    parser.add_argument('--model_summary_interval', type=int, default=10)
    parser.add_argument('--model_summary_num_batches', type=int, default=50)
    parser.add_argument('--checkpoint_interval', type=int, default=50)
    parser.add_argument('--keep_checkpoint_every_n_hours', type=float, default=2)
    parser.add_argument('--keep_n_last_checkpoints', type=int, default=5)
    parser.add_argument('--max_checkpoint_save_trials', type=int, default=5)

    # Resource allocation
    parser.add_argument('--intra_op_parallelism', type=int, default=1)
    parser.add_argument('--inter_op_parallelism', type=int, default=4)
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--cpu_train_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--cpu_test_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--cpu_train_queue_min_after_dequeue', type=int, default=2000)
    parser.add_argument('--cpu_test_queue_min_after_dequeue', type=int, default=0)
    parser.add_argument('--cpu_train_queue_threads', type=int, default=4)
    parser.add_argument('--cpu_test_queue_threads', type=int, default=1)

    # Train test split
    parser.add_argument('--train_test_split_ratio', type=float, default=4.0)
    parser.add_argument('--train_test_split_shuffle', action="store_true")
    parser.add_argument('--max_num_train_files', type=int, default=-1)
    parser.add_argument('--max_num_test_files', type=int, default=-1)

    # Learning parameters
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--max_grad_global_norm', type=float, default=1e3)
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_decay_epochs', type=int, default=10)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.96)
    parser.add_argument('--learning_rate_decay_staircase', type=argparse_bool, default=False)

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
    parser.add_argument('--maxpool_after_each_block', type=argparse_bool, default=True)
    parser.add_argument('--max_num_blocks', type=int, default=-1)
    parser.add_argument('--max_output_grid_size', type=int, default=8)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--add_bias_3dconv', type=argparse_bool, default=False)
    parser.add_argument('--use_batch_norm_3dconv', type=argparse_bool, default=False)
    parser.add_argument('--use_batch_norm_regression', type=argparse_bool, default=False)
    parser.add_argument('--activation_fn_3dconv', type=str, default="relu")
    parser.add_argument('--num_units_regression', type=str, default="1024")
    parser.add_argument('--activation_fn_regression', type=str, default="relu")

    args = parser.parse_args()

    run(args)
