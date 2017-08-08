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
import yaml
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.memory_stats as tf_memory_stats
import tf_utils
import data_provider
from attribute_dict import AttributeDict
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


def compute_dataset_stats(batches_generator):
    data_size = 0
    sums = None
    sq_sums = None
    for batches in batches_generator:
        if sums is None:
            sums = [np.zeros(batch.shape[1:]) for batch in batches]
            sq_sums = [np.zeros(batch.shape[1:]) for batch in batches]
        for i, batch in enumerate(batches):
            sums[i] += np.sum(batch, axis=0)
            sq_sums[i] += np.sum(np.square(batch), axis=0)
            assert(batch.shape[0] == batches[0].shape[0])
        data_size += batches[0].shape[0]
    means = []
    stddevs = []
    for i in xrange(len(sums)):
        mean = sums[i] / data_size
        stddev = (sq_sums[i] - np.square(sums[i]) / data_size) / (data_size - 1)
        stddev[np.abs(stddev) < 1e-5] = 1
        stddev = np.sqrt(stddev)
        assert(np.all(np.isfinite(mean)))
        assert(np.all(np.isfinite(stddev)))
        means.append(mean)
        stddevs.append(stddev)
    return data_size, zip(means, stddevs)


def compute_dataset_stats_from_hdf5_files_v3(filenames, field_names, compute_z_scores=True):
    assert(len(filenames) > 0)

    def batches_generator(filenames, field_names):
        for filename in filenames:
            batches = []
            record_batch = data_record.read_hdf5_records_v3(filename)
            for name in field_names:
                batch = getattr(record_batch, name)
                batches.append(batch)
            yield batches
    data_size, stats = compute_dataset_stats(batches_generator(filenames, field_names))
    if compute_z_scores:

        def z_score_batches_generator(filenames, field_names):
            for filename in filenames:
                batches = []
                record_batch = data_record.read_hdf5_records_v3(filename)
                for i, name in enumerate(field_names):
                    batch = getattr(record_batch, name)
                    mean = stats[i][0]
                    stddev = stats[i][1]
                    z_score = (batch - mean[np.newaxis, ...]) / stddev[np.newaxis, ...]
                    batches.append(z_score)
                yield batches
        _, z_score_stats = compute_dataset_stats(z_score_batches_generator(filenames, field_names))
        all_stats = []
        for i in xrange(len(stats)):
            all_stats.append(stats[i])
            all_stats.append(z_score_stats[i])
        stats = all_stats
    return data_size, stats


def get_input_normalization(filenames, input_stats_filename):
    # Mean and stddev of input for normalization
    if os.path.isfile(input_stats_filename):
        numpy_dict = data_record.read_hdf5_file_to_numpy_dict(input_stats_filename)
        all_data_size = int(numpy_dict["all_data_size"])
        mean_in_grid_3d_np = numpy_dict["mean_in_grid_3d"]
        stddev_in_grid_3d_np = numpy_dict["stddev_in_grid_3d"]
        mean_z_score = numpy_dict["mean_z_score"]
        stddev_z_score = numpy_dict["stddev_z_score"]
        tmp_record_batch = data_record.read_hdf5_records_v3(filenames[0])
        assert(np.all(numpy_dict["mean_in_grid_3d"].shape == tmp_record_batch.in_grid_3ds.shape[1:]))
    else:
        print("Computing data statistics")
        all_data_size, ((mean_in_grid_3d_np, stddev_in_grid_3d_np), (mean_z_score, stddev_z_score)) \
            = compute_dataset_stats_from_hdf5_files_v3(filenames, ["in_grid_3ds"], compute_z_scores=True)
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
            "mean_in_grid_3d": mean_in_grid_3d_np,
            "stddev_in_grid_3d": stddev_in_grid_3d_np,
            "mean_z_score": mean_z_score,
            "stddev_z_score": stddev_z_score,
        })
    print("Data statistics:")
    print("  Mean of in_grid_3d:", np.mean(mean_in_grid_3d_np.flatten()))
    print("  Stddev of in_grid_3d:", np.mean(stddev_in_grid_3d_np.flatten()))
    print("  Mean of z_score:", np.mean(mean_z_score.flatten()))
    print("  Stddev of z_score:", np.mean(stddev_z_score.flatten()))
    print("  Size of full dataset:", all_data_size)

    return mean_in_grid_3d_np, stddev_in_grid_3d_np


def save_model(sess, saver, store_path, global_step_tf, max_trials, retry_save_wait_time=5, verbose=False):
    saved = False
    trials = 0
    while not saved and trials < max_trials:
        try:
            trials += 1
            timer = Timer()
            saver.save(sess, store_path, global_step=global_step_tf)
            save_time = timer.restart()
            saved = True
            if verbose:
                print("Saving took {} s".format(save_time))
        except Exception, err:
            print("ERROR: Exception when trying to save model: {}".format(err))
            if trials < max_trials:
                if verbose:
                    print("Retrying to save model in {} s...".format(retry_save_wait_time))
                time.sleep(retry_save_wait_time)
            else:
                raise


def update_config_from_cmdline(config, args):
    if "tensorflow" not in config:
        config["tensorflow"] = {}
    config["tensorflow"]["gpu_ids"] = args.gpu_ids
    config["tensorflow"]["strict_devices"] = args.strict_devices
    config["tensorflow"]["multi_gpu_ps_id"] = args.multi_gpu_ps_id
    config["tensorflow"]["gpu_memory_fraction"] = args.gpu_memory_fraction
    config["tensorflow"]["intra_op_parallelism"] = args.intra_op_parallelism
    config["tensorflow"]["inter_op_parallelism"] = args.inter_op_parallelism
    config["tensorflow"]["cpu_train_queue_capacity"] = args.cpu_train_queue_capacity
    config["tensorflow"]["cpu_test_queue_capacity"] = args.cpu_test_queue_capacity
    config["tensorflow"]["cpu_train_queue_min_after_dequeue"] = args.cpu_train_queue_min_after_dequeue
    config["tensorflow"]["cpu_test_queue_min_after_dequeue"] = args.cpu_test_queue_min_after_dequeue
    config["tensorflow"]["cpu_train_queue_threads"] = args.cpu_train_queue_threads
    config["tensorflow"]["cpu_test_queue_threads"] = args.cpu_test_queue_threads
    config["tensorflow"]["log_device_placement"] = args.log_device_placement
    config["tensorflow"]["create_tf_timeline"] = args.create_tf_timeline
    if "io" not in config:
        config["io"] = {}
    config["io"]["async_timeout"] = args.async_timeout
    config["io"]["validation_interval"] = args.validation_interval
    config["io"]["train_summary_interval"] = args.train_summary_interval
    config["io"]["model_summary_interval"] = args.model_summary_interval
    config["io"]["model_summary_num_batches"] = args.model_summary_num_batches
    config["io"]["checkpoint_interval"] = args.checkpoint_interval
    config["io"]["keep_checkpoint_every_n_hours"] = args.keep_checkpoint_every_n_hours
    config["io"]["keep_n_last_checkpoints"] = args.keep_n_last_checkpoints
    config["io"]["max_checkpoint_save_trials"] = args.max_checkpoint_save_trials
    if "training" not in config:
        config["training"] = {}
    config["training"]["num_epochs"] = args.num_epochs
    config["training"]["batch_size"] = args.batch_size
    config["training"]["optimizer"] = args.optimizer
    config["training"]["max_grad_global_norm"] = args.max_grad_global_norm
    config["training"]["initial_learning_rate"] = args.initial_learning_rate
    config["training"]["learning_rate_decay_epochs"] = args.learning_rate_decay_epochs
    config["training"]["learning_rate_decay_rate"] = args.learning_rate_decay_rate
    config["training"]["learning_rate_decay_staircase"] = args.learning_rate_decay_staircase
    if "data" not in config:
        config["data"] = {}
    config["data"]["train_test_split_ratio"] = args.train_test_split_ratio
    config["data"]["train_test_split_shuffle"] = args.train_test_split_shuffle
    config["data"]["max_num_train_files"] = args.max_num_train_files
    config["data"]["max_num_test_files"] = args.max_num_test_files
    config["data"]["fake_constant_data"] = args.fake_constant_data
    config["data"]["fake_random_data"] = args.fake_random_data
    config["data"]["input_id"] = args.input_id
    config["data"]["target_id"] = args.target_id
    config["data"]["input_stats_filename"] = args.input_stats_filename
    config["data"]["obs_levels_to_use"] = args.obs_levels_to_use
    config["data"]["subvolume_slice_x"] = args.subvolume_slice_x
    config["data"]["subvolume_slice_y"] = args.subvolume_slice_y
    config["data"]["subvolume_slice_z"] = args.subvolume_slice_z
    config["data"]["normalize_input"] = args.normalize_input
    config["data"]["normalize_target"] = args.normalize_target
    return config


def get_default_model_config():
    config = {}
    config["modules"] = ["conv3d", "regression"]
    config["conv3d"] = {}
    config["conv3d"]["num_convs_per_block"] = 8
    config["conv3d"]["initial_num_filters"] = 8
    config["conv3d"]["filter_increase_per_block"] = 0
    config["conv3d"]["filter_increase_within_block"] = 6
    config["conv3d"]["maxpool_after_each_block"] = False
    config["conv3d"]["max_num_blocks"] = -1
    config["conv3d"]["max_output_grid_size"] = 8
    config["conv3d"]["dropout_rate"] = 0.5
    config["conv3d"]["add_bias"] = False
    config["conv3d"]["use_batch_norm"] = False
    config["conv3d"]["activation_fn"] = "relu"
    config["regression"] = {}
    config["regression"]["use_batch_norm"] = False
    config["regression"]["num_units"] = "1024"
    config["regression"]["activation_fn"] = "relu"
    config["upsampling"] = {}
    config["upsampling"]["num_convs_per_block"] = 4
    config["upsampling"]["add_bias"] = True
    config["upsampling"]["use_batch_norm"] = False
    config["upsampling"]["filter_decrease_per_block"] = 8
    config["upsampling"]["filter_decrease_within_block"] = 16
    config["upsampling"]["activation_fn"] = "relu"
    return config


def run(args):
    # Read config file
    if args.config is None:
        cfg = {}
        update_config_from_cmdline(cfg, args)
    else:
        with file(args.config, "r") as config_file:
            cfg = yaml.load(config_file)
    # TODO: Make command line override file config

    # Read model config
    if "model" in cfg:
        model_config = cfg["model"]
        if args.model_config is None:
            print("ERROR: Model configuration must be in general config file or provided in extra config file.")
            import sys
            sys.exit(1)
    else:
        model_config = {}

    with file(args.model_config, "r") as config_file:
        tmp_model_config = yaml.load(config_file)
        model_config.update(tmp_model_config)

    cfg = AttributeDict.convert_deep(cfg)
    model_config = AttributeDict.convert_deep(model_config)

    cfg.io.retry_save_wait_time = 3
    cfg.io.max_num_summary_errors = 3
    if cfg.io.model_summary_num_batches <= 0:
        cfg.io.model_summary_num_batches = np.iinfo(np.int32).max

    # Where to save checkpoints
    model_store_path = os.path.join(args.store_path, "model")

    # Learning parameters
    num_epochs = cfg.training.num_epochs
    batch_size = cfg.training.batch_size

    # Determine train/test dataset filenames and optionally split them
    split_data = args.test_data_path is None
    if split_data:
        train_percentage = cfg.data.train_test_split_ratio / (1 + cfg.data.train_test_split_ratio)
        print("Train percentage: {}".format(train_percentage))
        filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
        filenames = sorted(list(filename_generator))
        if cfg.data.train_test_split_shuffle:
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
    if cfg.data.max_num_train_files > 0:
        train_filenames = train_filenames[:cfg.data.max_num_train_files]
        print("Using {} train dataset files".format(len(train_filenames)))
    else:
        print("Using all train dataset files")
    if cfg.data.max_num_test_files > 0:
        test_filenames = test_filenames[:cfg.data.max_num_test_files]
        print("Using {} test dataset files".format(len(test_filenames)))
    else:
        print("Using all test dataset files")

    # Input configuration, i.e. which slices and channels from the 3D grids to use.
    # First read any of the data records
    tmp_record_batch = data_record.read_hdf5_records_v3(train_filenames[0])
    assert(tmp_record_batch.in_grid_3ds.shape[0] > 0)
    assert(np.all(tmp_record_batch.in_grid_3ds.shape == tmp_record_batch.out_grid_3ds.shape))
    tmp_records = list(data_record.generate_single_records_from_batch_v3(tmp_record_batch))
    tmp_record = tmp_records[0]
    print("Observation levels in data record: {}".format(tmp_record.obs_levels))
    raw_in_grid_3d = tmp_record.in_grid_3d
    # Determine subvolume slices
    if cfg.data.subvolume_slice_x is None:
        subvolume_slice_x = slice(0, raw_in_grid_3d.shape[0])
    else:
        subvolume_slice_x = slice(*[int(x) for x in cfg.data.subvolume_slice_x.split(',')])
    if cfg.data.subvolume_slice_y is None:
        subvolume_slice_y = slice(0, raw_in_grid_3d.shape[1])
    else:
        subvolume_slice_y = slice(*[int(x) for x in cfg.data.subvolume_slice_y.split(',')])
    if cfg.data.subvolume_slice_z is None:
        subvolume_slice_z = slice(0, raw_in_grid_3d.shape[2])
    else:
        subvolume_slice_z = slice(*[int(x) for x in cfg.data.subvolume_slice_z.split(',')])
    # Print used subvolume slices and channels
    print("Subvolume slice x: {}".format(subvolume_slice_x))
    print("subvolume slice y: {}".format(subvolume_slice_y))
    print("subvolume slice z: {}".format(subvolume_slice_z))
    print("Input id: {}".format(cfg.data.target_id))
    print("Target id: {}".format(cfg.data.target_id))

    # Retrieval functions for input from data records
    if cfg.data.input_id == "in_grid_3d":
        # Determine channels to use
        if cfg.data.obs_levels_to_use is None:
            in_grid_3d_channels = range(raw_in_grid_3d.shape[-1])
        else:
            obs_levels_to_use = [int(x) for x in cfg.data.obs_levels_to_use.split(',')]
            in_grid_3d_channels = []
            for level in obs_levels_to_use:
                in_grid_3d_channels.append(2 * level)
                in_grid_3d_channels.append(2 * level + 1)
        print("Channels of in_grid_3d: {}".format(in_grid_3d_channels))
        def get_input_from_record(record):
            return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    elif cfg.data.input_id.startswith("in_grid_3d["):
        in_channels = cfg.data.input_id[len("in_grid_3d"):]
        in_channels = [int(x) for x in in_channels.strip("[]").split(",")]
        def get_input_from_record(record):
            return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_channels]
    elif cfg.data.input_id.startswith("rand["):
        shape = cfg.data.input_id[len("rand"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]
        def get_input_from_record(record):
            return np.random.rand(*shape)
    elif cfg.data.input_id.startswith("randn["):
        shape = cfg.data.input_id[len("randn"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]
        def get_input_from_record(record):
            return np.random.randn(*shape)
    else:
        raise NotImplementedError("Unknown input name: {}".format(cfg.data.input_id))

    if cfg.data.normalize_input:
        input_stats_filename = cfg.data.input_stats_filename
        if not cfg.data.input_stats_filename:
            input_stats_filename = os.path.join(args.data_path, file_helpers.DEFAULT_HDF5_STATS_FILENAME)
        mean_in_grid_3d_np, stddev_in_grid_3d_np = get_input_normalization(all_filenames, input_stats_filename)
        mean_record = data_record.RecordV3(None, mean_in_grid_3d_np, None, None, None)
        mean_input_np = get_input_from_record(mean_record)
        stddev_record = data_record.RecordV3(None, stddev_in_grid_3d_np, None, None, None)
        stddev_input_np = get_input_from_record(stddev_record)
        print("  mean_input_np.shape", mean_input_np.shape)
        print("  stddev_input_np.shape", stddev_input_np.shape)

        get_unnormalized_input_from_record = get_input_from_record

        # Retrieval functions for normalized input
        def get_input_from_record(record):
            single_input = get_unnormalized_input_from_record(record)
            if cfg.data.normalize_input:
                single_input = (single_input - mean_input_np) / stddev_input_np
            return single_input

    # Retrieval functions for target from data records
    if cfg.data.target_id == "reward":
        def get_target_from_record(record):
            return record.rewards[..., 0].reshape((1,))
    elif cfg.data.target_id == "norm_reward":
        def get_target_from_record(record):
            return record.rewards[..., 1].reshape((1,))
    elif cfg.data.target_id == "prob_reward":
        def get_target_from_record(record):
            return record.rewards[..., 2].reshape((1,))
    elif cfg.data.target_id == "norm_prob_reward":
        def get_target_from_record(record):
            return record.rewards[..., 3].reshape((1,))
    elif cfg.data.target_id == "score":
        def get_target_from_record(record):
            return record.scores[..., 0].reshape((1,))
    elif cfg.data.target_id == "norm_score":
        def get_target_from_record(record):
            return record.scores[..., 1].reshape((1,))
    elif cfg.data.target_id == "prob_score":
        def get_target_from_record(record):
            return record.scores[..., 2].reshape((1,))
    elif cfg.data.target_id == "norm_prob_score":
        def get_target_from_record(record):
            return record.scores[..., 3].reshape((1,))
    elif cfg.data.target_id == "mean_occupancy":
        def get_target_from_record(record):
            return np.mean(record.in_grid_3d[..., 0::2]).reshape((1,))
    elif cfg.data.target_id == "sum_occupancy":
        def get_target_from_record(record):
            return np.sum(record.in_grid_3d[..., 0::2]).reshape((1,))
    elif cfg.data.target_id == "mean_observation":
        def get_target_from_record(record):
            return np.mean(record.in_grid_3d[..., 1::2]).reshape((1,))
    elif cfg.data.target_id == "sum_observation":
        def get_target_from_record(record):
            return np.sum(record.in_grid_3d[..., 1::2]).reshape((1,))
    elif cfg.data.target_id == "in_grid_3d":
        assert(cfg.data.input_id == "in_grid_3d")
        def get_target_from_record(record):
            return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    elif cfg.data.target_id == "norm_in_grid_3d":
        assert(cfg.data.input_id == "in_grid_3d")
        def get_target_from_record(record):
            single_grid_3d = record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
            if cfg.data.normalize_input:
                single_grid_3d = (single_grid_3d - mean_input_np) / stddev_input_np
            return single_grid_3d
    elif cfg.data.target_id == "out_grid_3d":
        assert(cfg.data.input_id == "in_grid_3d")
        def get_target_from_record(record):
            return record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    elif cfg.data.target_id == "norm_out_grid_3d":
        assert(cfg.data.input_id == "in_grid_3d")
        def get_target_from_record(record):
            single_grid_3d = record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
            if cfg.data.normalize_input:
                single_grid_3d = (single_grid_3d - mean_input_np) / stddev_input_np
            return single_grid_3d
    elif cfg.data.target_id.startswith("out_grid_3d["):
        out_channels = cfg.data.target_id[len("out_grid_3d"):]
        out_channels = [int(x) for x in out_channels.strip("[]").split(",")]
        def get_target_from_record(record):
            return record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, out_channels]
    elif cfg.data.target_id == "input":
        def get_target_from_record(record):
            return get_input_from_record(record)
    elif cfg.data.target_id.startswith("rand["):
        shape = cfg.data.target_id[len("rand"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]
        def get_target_from_record(record):
            return np.random.rand(*shape)
    elif cfg.data.target_id.startswith("randn["):
        shape = cfg.data.target_id[len("randn"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]
        def get_target_from_record(record):
            return np.random.randn(*shape)
    else:
        raise NotImplementedError("Unknown target name: {}".format(cfg.data.target_id))

    if cfg.data.normalize_target:
        print("Computing target statistics. This can take a while.")

        def target_batches_generator(filenames):
            for filename in filenames:
                for record in data_record.generate_single_records_from_hdf5_file_v3(filename):
                    single_target = get_target_from_record(record)
                    target_batch = single_target[np.newaxis, ...]
                    yield [target_batch]

        _, ((mean_target_np, stddev_target_np),) \
            = compute_dataset_stats(target_batches_generator(all_filenames))
        print("Target data statistics:")
        print("  Mean of target:", np.mean(mean_target_np.flatten()))
        print("  Stddev of target:", np.mean(stddev_target_np.flatten()))

        get_unnormalized_target_from_record = get_target_from_record

        # Retrieval functions for normalized target
        def get_target_from_record(record):
            single_target = get_unnormalized_target_from_record(record)
            if cfg.data.normalize_target:
                single_target = (single_target - mean_target_np) / stddev_target_np
            return single_target

    # Report some stats on input and outputs for the first data file
    # This is only for sanity checking
    # TODO: This can be confusing as we just take mean and average over all 3d positions
    if args.verbose:
        print("Stats on inputs and outputs")
        tmp_inputs = [get_input_from_record(record) for record in tmp_records]
        for i in xrange(tmp_inputs[0].shape[-1]):
            values = [input[..., i] for input in tmp_inputs]
            print("  Mean of input {}: {}".format(i, np.mean(values)))
            print("  Stddev of input {}: {}".format(i, np.std(values)))
            print("  Min of input {}: {}".format(i, np.min(values)))
            print("  Max of input {}: {}".format(i, np.max(values)))
        tmp_targets = [get_target_from_record(record) for record in tmp_records]
        if len(tmp_targets[0].shape) > 1:
            for i in xrange(tmp_targets[0].shape[-1]):
                values = [target[..., i] for target in tmp_targets]
                print("  Mean of target {}: {}".format(i, np.mean(values)))
                print("  Stddev of target {}: {}".format(i, np.std(values)))
                print("  Min of target {}: {}".format(i, np.min(values)))
                print("  Max of target {}: {}".format(i, np.max(values)))
        values = [target for target in tmp_targets]
        print("  Mean of target: {}".format(np.mean(values)))
        print("  Stddev of target: {}".format(np.std(values)))
        print("  Min of target: {}".format(np.min(values)))
        print("  Max of target: {}".format(np.max(values)))

    # Retrieve input and target shapes
    in_grid_3d = tmp_record.in_grid_3d
    in_grid_3d_shape = list(in_grid_3d.shape)
    input_shape = list(get_input_from_record(tmp_record).shape)
    target_shape = list(get_target_from_record(tmp_record).shape)
    print("Input and target shapes:")
    print("  Shape of grid_3d: {}".format(in_grid_3d_shape))
    print("  Shape of input: {}".format(input_shape))
    print("  Shape of target: {}".format(target_shape))

    # Setup TF step and epoch counters
    global_step_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='global_step')
    epoch_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='epoch')
    inc_epoch = epoch_tf.assign_add(tf.constant(1, dtype=tf.int64))

    # Configure tensorflow
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = cfg.tensorflow.gpu_memory_fraction
    tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
    tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
    tf_config.log_device_placement = cfg.tensorflow.log_device_placement

    # Multi-device configuration
    gpu_ids = cfg.tensorflow.gpu_ids
    if gpu_ids is None:
        gpu_ids = tf_utils.get_available_gpu_ids()
    else:
        gpu_ids = [int(s) for s in gpu_ids.strip("[]").split(",")]
    assert(len(gpu_ids) >= 1)

    use_multi_gpu = len(gpu_ids) > 1
    if use_multi_gpu:
        variables_on_cpu = False
        assert(all([gpu_id >= 0 for gpu_id in gpu_ids]))
        device_names = [tf_utils.gpu_device_name(gpu_id) for gpu_id in gpu_ids]
    else:
        variables_on_cpu = False
        device_names = [tf_utils.tf_device_name(gpu_ids[0])]

    existing_device_names = tf_utils.get_available_device_names()
    filtered_device_names = filter(lambda x: x in existing_device_names, device_names)
    if len(filtered_device_names) < len(device_names):
        print("ERROR: Not all desired devices are available.")
        print("Available devices:")
        for device_name in existing_device_names:
            print("  {}".format(device_name))
        print("Filtered devices:")
        for device_name in filtered_device_names:
            print("  {}".format(device_name))
        if cfg.tensorflow.strict_devices:
            import sys
            sys.exit(1)
        print("Continuing with filtered devices")
        time.sleep(5)
        # resp = raw_input("Continue with filtered devices? [y/n] ")
        # if resp != "y":
        #     import sys
        #     sys.exit(1)
        device_names = filtered_device_names

    if use_multi_gpu:
        if cfg.tensorflow.multi_gpu_ps_id is None:
            ps_device_name = device_names[0]
        else:
            ps_device_name = tf_utils.tf_device_name(cfg.tensorflow.multi_gpu_ps_id)
    else:
        ps_device_name = None

    batch_size_per_device = batch_size / len(device_names)
    if not batch_size % batch_size_per_device == 0:
        print("Batch size has to be equally divideable by number of devices")

    print("Used devices:")
    for device_name in device_names:
        print("  {}".format(device_name))
    if use_multi_gpu:
        print("Parameter server device: {}".format(ps_device_name))
    print("Total batch size: {}, per device batch size: {}".format(batch_size, batch_size_per_device))

    sess = tf.Session(config=tf_config)
    coord = tf.train.Coordinator()

    class InputPipeline(object):

        def _record_provider_factory(self, hdf5_input_pipeline):
            def record_provider():
                if cfg.data.fake_constant_data:
                    single_input = np.ones(input_shape)
                    single_target = np.ones(target_shape)
                elif cfg.data.fake_random_data:
                    single_input = np.random.randn(input_shape)
                    single_target = np.random.randn(target_shape)
                else:
                    record = hdf5_input_pipeline.get_next_record()
                    single_input = get_input_from_record(record)
                    single_target = get_target_from_record(record)
                assert(np.all(np.isfinite(single_input)))
                assert(np.all(np.isfinite(single_target)))
                return single_input, single_target
            return record_provider

        def __init__(self, filenames, queue_capacity, min_after_dequeue, shuffle, num_threads, name):
            # Create HDF5 readers
            self._hdf5_input_pipeline = data_record.HDF5ReaderProcessCoordinator(
                filenames, coord, shuffle=shuffle, hdf5_record_version=data_record.HDF5_RECORD_VERSION_3,
                timeout=cfg.io.async_timeout, num_processes=num_threads, verbose=args.verbose >= 2)
            self._num_records = None

            tensor_dtypes = [tf.float32, tf.float32]
            tensor_shapes = [input_shape, target_shape]
            self._tf_pipeline = data_provider.TFInputPipeline(
                self._record_provider_factory(self._hdf5_input_pipeline),
                sess, coord, batch_size_per_device, tensor_shapes, tensor_dtypes,
                queue_capacity=queue_capacity,
                min_after_dequeue=min_after_dequeue,
                shuffle=shuffle,
                num_threads=num_threads,
                timeout=cfg.io.async_timeout,
                name="{}_tf_input_pipeline".format(name),
                verbose=args.verbose >= 2)

            # Retrieve tensors from data bridge
            self._input_batch, self._target_batch = self._tf_pipeline.tensors

        def start(self):
            self._hdf5_input_pipeline.start()
            self._tf_pipeline.start()

        @property
        def num_records(self):
            if self._num_records is None:
                self._num_records = self._hdf5_input_pipeline.compute_num_records()
            return self._num_records

        @property
        def tensors(self):
            return self._tf_pipeline.tensors

        @property
        def input_batch(self):
            return self._input_batch

        @property
        def target_batch(self):
            return self._target_batch

        @property
        def threads(self):
            return [self._hdf5_input_pipeline.thread] + self._tf_pipeline.threads

    with tf.device("/cpu:0"):
        train_input_pipeline = InputPipeline(
            train_filenames, cfg.tensorflow.cpu_train_queue_capacity, cfg.tensorflow.cpu_train_queue_min_after_dequeue,
            shuffle=True, num_threads=cfg.tensorflow.cpu_train_queue_threads, name="train")
        test_input_pipeline = InputPipeline(
            test_filenames, cfg.tensorflow.cpu_test_queue_capacity, cfg.tensorflow.cpu_test_queue_min_after_dequeue,
            shuffle=True, num_threads=cfg.tensorflow.cpu_test_queue_threads, name="test")
        print("# records in train dataset: {}".format(train_input_pipeline.num_records))
        print("# records in test dataset: {}".format(test_input_pipeline.num_records))

    # Create model

    train_staging_areas = []
    train_models = []
    test_staging_areas = []
    test_models = []

    for i, device_name in enumerate(device_names):
        with tf.device(device_name):
            # input_batch = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name="in_input")
            reuse = i > 0
            with tf.variable_scope("model", reuse=reuse):
                staging_area = data_provider.TFStagingArea(train_input_pipeline.tensors, device_name)
                train_staging_areas.append(staging_area)
                model = models.Model(model_config,
                                     staging_area.tensors[0],
                                     staging_area.tensors[1],
                                     is_training=True,
                                     variables_on_cpu=variables_on_cpu,
                                     verbose=args.verbose)
                train_models.append(model)
            # Generate output and loss function for testing (no dropout)
            with tf.variable_scope("model", reuse=True):
                staging_area = data_provider.TFStagingArea(test_input_pipeline.tensors, device_name)
                test_staging_areas.append(staging_area)
                model = models.Model(model_config,
                                     staging_area.tensors[0],
                                     staging_area.tensors[1],
                                     is_training=False,
                                     variables_on_cpu=variables_on_cpu,
                                     verbose=args.verbose)
                test_models.append(model)

    # Generate ground-truth inputs for computing loss function
    # train_target_batch = tf.placeholder(dtype=np.float32, shape=[None], name="target_batch")

    with tf.device(ps_device_name):
        if use_multi_gpu:
            # Wrap multiple models on GPUs into single model
            train_model = models.MultiGpuModelWrapper(train_models, args.verbose >= 2)
            test_model = models.MultiGpuModelWrapper(test_models, args.verbose >= 2)
        else:
            train_model = train_models[0]
            test_model = test_models[0]

        # Create optimizer
        optimizer_class = tf_utils.get_optimizer_by_name(cfg.training.optimizer, tf.train.AdamOptimizer)
        learning_rate_tf = tf.train.exponential_decay(cfg.training.initial_learning_rate,
                                                      epoch_tf,
                                                      cfg.training.learning_rate_decay_epochs,
                                                      cfg.training.learning_rate_decay_rate,
                                                      cfg.training.learning_rate_decay_staircase)
        opt = optimizer_class(learning_rate_tf)

        # Get variables and gradients
        variables = train_model.variables
        gradients = train_model.gradients
        gradients, _ = tf.clip_by_global_norm(gradients, cfg.training.max_grad_global_norm)
        gradients_and_variables = list(zip(gradients, variables))
        # gradients_and_variables = opt.compute_gradients(train_model.loss, variables)
        var_to_grad_dict = {var: grad for grad, var in gradients_and_variables}
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt.apply_gradients(gradients_and_variables, global_step=global_step_tf)

        var_global_norm = tf.global_norm(variables)
        grad_global_norm = tf.global_norm(gradients)

    with tf.device(tf_utils.cpu_device_name()):
        # Tensorboard summaries
        summary_loss = tf.placeholder(tf.float32, [])
        summary_loss_min = tf.placeholder(tf.float32, [])
        summary_loss_max = tf.placeholder(tf.float32, [])
        summary_grad_global_norm = tf.placeholder(tf.float32, [])
        summary_var_global_norm = tf.placeholder(tf.float32, [])
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

    if args.verbose:
        print("Tensorflow variables:")
        for var in tf.global_variables():
            print("  {}: {}".format(var.name, var.shape))
    print("Model variables:")
    for grad, var in gradients_and_variables:
        print("  {}: {}".format(var.name, var.shape))

    with tf.device(tf_utils.cpu_device_name()):
        # Model histogram summaries
        if cfg.io.model_summary_interval > 0:
            with tf.device("/cpu:0"):

                def get_model_hist_summary(input_pipeline, model):
                    model_summary_dict = {}
                    if len(target_shape) > 1:
                        for i in xrange(target_shape[-1]):
                            model_summary_dict["target_batch/{}".format(i)] \
                                = input_pipeline.target_batch[..., slice(i, i+1)]
                        model_summary_dict["target_batch/all"] = input_pipeline.target_batch
                    else:
                        model_summary_dict["target_batch"] = input_pipeline.target_batch
                    for i in xrange(input_shape[-1]):
                        model_summary_dict["input_batch/{}".format(i)] \
                            = input_pipeline.input_batch[..., slice(i, i+1)]
                    model_summary_dict["input_batch/all"] = input_pipeline.input_batch
                    for name, tensor in model.summaries.iteritems():
                        model_summary_dict["model/" + name] = tensor
                        # Reuse existing gradient expressions
                        if tensor in var_to_grad_dict:
                            grad = var_to_grad_dict[tensor]
                            if grad is not None:
                                model_summary_dict["model/" + name + "_grad"] = grad
                    model_hist_summary = tf_utils.ModelHistogramSummary(model_summary_dict)
                    return model_hist_summary

                train_model_hist_summary = get_model_hist_summary(train_input_pipeline, train_model)

    saver = tf.train.Saver(max_to_keep=cfg.io.keep_n_last_checkpoints,
                           keep_checkpoint_every_n_hours=cfg.io.keep_checkpoint_every_n_hours)

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
        print("Model variables: {} ({} MB)".format(model_size, model_size / 1024. / 1024.))
        print("Model gradients: {} ({} MB)".format(model_grad_size, model_grad_size / 1024. / 1024.))
        for name, module in train_model.modules.iteritems():
            module_size = np.sum([tf_utils.variable_size(var) for var in module.variables])
            print("Module {} variables: {} ({} MB)".format(name, module_size, module_size / 1024. / 1024.))

        # Initialize tensorflow session
        if cfg.tensorflow.create_tf_timeline:
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
                response = raw_input("WARNING: No previous checkpoint found. Continue? [y/n] ")
                if response != "y":
                    raise RuntimeError("Could not find previous checkpoint")
            else:
                print('Found previous checkpoint... restoring')
                saver.restore(sess, ckpt.model_checkpoint_path)

        # Create timelines for profiling?
        if cfg.tensorflow.create_tf_timeline:
            train_options = tf.RunOptions(timeout_in_ms=100000,
                                          trace_level=tf.RunOptions.FULL_TRACE)
        else:
            train_options = None

        # Get preload ops and make sure the GPU pipeline is filled with a mini-batch
        train_preload_ops = [staging_area.preload_op for staging_area in train_staging_areas]
        test_preload_ops = [staging_area.preload_op for staging_area in test_staging_areas]
        train_op = tf.group(train_op, *train_preload_ops)

        # Report memory usage for first device. Should be same for all.
        with tf.device(device_names[0]):
            max_memory_gpu = tf_memory_stats.BytesLimit()
            peak_use_memory_gpu = tf_memory_stats.MaxBytesInUse()

        sess.graph.finalize()

        print("Preloading GPU")
        sess.run(train_preload_ops + test_preload_ops)

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

            do_summary = cfg.io.train_summary_interval > 0 and epoch % cfg.io.train_summary_interval == 0
            if do_summary:
                var_global_norm_v = 0.0
                grad_global_norm_v = 0.0
                if args.verbose:
                    print("Generating train summary")

            do_model_summary = cfg.io.model_summary_interval > 0 and epoch % cfg.io.model_summary_interval == 0
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
                fetches = [train_op, train_model.loss, train_model.loss_min, train_model.loss_max]

                if do_summary:
                    summary_offset = len(fetches)
                    fetches.extend([var_global_norm, grad_global_norm])

                if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
                    model_summary_offset = len(fetches)
                    fetches.extend(train_model_hist_summary.fetches)

                fetched = \
                    sess.run(fetches,
                             options=train_options,
                             run_metadata=run_metadata)
                _, loss_v, loss_min_v, loss_max_v = fetched[:4]

                if do_summary:
                    var_global_norm_v += fetched[summary_offset]
                    grad_global_norm_v += fetched[summary_offset + 1]

                if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
                    for i, value in enumerate(fetched[model_summary_offset:]):
                        # Make sure we copy the model summary tensors (otherwise it might be pinned GPU memory)
                        model_summary_fetched[i].append(np.array(value))

                if cfg.tensorflow.create_tf_timeline:
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
                    if num_summary_errors >= cfg.io.max_num_summary_errors:
                        print("Too many summary errors occured. Aborting.")
                        raise
            if do_model_summary:
                # TODO: Build model summary in separate thread?
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
                    if num_summary_errors >= cfg.io.max_num_summary_errors:
                        print("Too many summary errors occured. Aborting.")
                        raise
                del model_summary_fetched
                del feed_dict
                del model_summary

            print("train result:")
            print("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
                epoch, total_loss_value, total_loss_min, total_loss_max))
            epoch_time_sec = timer.restart()
            epoch_time_min = epoch_time_sec / 60.
            print("train stats:")
            print("  batches: {}, records: {}, time: {} min".format(
                batch_count, record_count, epoch_time_min))
            record_per_sec = record_count / float(epoch_time_sec)
            batch_per_sec = batch_count / float(epoch_time_sec)
            print("  record/s: {}, batch/s: {}".format(record_per_sec, batch_per_sec))
            print("  total batches: {}, total records: {}".format(
                total_batch_count, total_record_count))

            # Heartbeat signal for Philly cluster
            progress = 100 * float(epoch - initial_epoch) / (num_epochs - initial_epoch)
            print("PROGRESS: {:05.2f}%".format(progress))

            sess.run([inc_epoch])
            learning_rate = float(sess.run([learning_rate_tf])[0])
            print("Current learning rate: {:e}".format(learning_rate))

            if cfg.io.validation_interval > 0 \
                    and (epoch + 1) % cfg.io.validation_interval == 0:
                # Perform validation
                total_loss_value = 0.0
                total_loss_min = +np.finfo(np.float32).max
                total_loss_max = -np.finfo(np.float32).max
                batch_count = 0
                for record_count in xrange(0, test_input_pipeline.num_records, batch_size):
                    _, loss_v, loss_min_v, loss_max_v = sess.run([
                        test_preload_ops,
                        test_model.loss, test_model.loss_min, test_model.loss_max])
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

            if epoch > 0 and epoch % cfg.io.checkpoint_interval == 0:
                print("Saving model at epoch {}".format(epoch))
                save_model(sess, saver, model_store_path, global_step_tf,
                           cfg.io.max_checkpoint_save_trials, cfg.io.retry_save_wait_time, verbose=True)

        print("Saving final model")
        save_model(sess, saver, model_store_path, global_step_tf,
                   cfg.io.max_checkpoint_save_trials, cfg.io.retry_save_wait_time, verbose=True)

    except Exception, exc:
        print("Exception in training loop: {}".format(exc))
        coord.request_stop(exc)
        raise exc
    finally:
        print("Requesting stop")
        coord.request_stop()
        coord.join(custom_threads, stop_grace_period_secs=cfg.io.async_timeout)


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
    parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--test-data-path', required=False, help='Test data path. If not provided data will be split.')
    parser.add_argument('--store-path', required=True, help='Store path.')
    parser.add_argument('--log-path', required=False, help='Log path.')
    parser.add_argument('--restore', type=argparse_bool, default=False, help='Whether to restore existing model.')
    parser.add_argument('--config', type=str, help='YAML configuration file.')
    parser.add_argument('--model-config', type=str, required=True, help='YAML description of model.')

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

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--gpu_ids', type=str,
                        help='GPU to be used (-1 for CPU). Comma separated list for multiple GPUs')
    parser.add_argument('--strict_devices', type=argparse_bool, default=False,
                        help='Should the user be prompted to use fewer devices instead of failing.')
    parser.add_argument('--multi_gpu_ps_id', type=int, help='Device to use as parameter server in Multi GPU mode')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--intra_op_parallelism', type=int, default=1)
    parser.add_argument('--inter_op_parallelism', type=int, default=4)
    parser.add_argument('--cpu_train_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--cpu_test_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--cpu_train_queue_min_after_dequeue', type=int, default=2000)
    parser.add_argument('--cpu_test_queue_min_after_dequeue', type=int, default=0)
    parser.add_argument('--cpu_train_queue_threads', type=int, default=4)
    parser.add_argument('--cpu_test_queue_threads', type=int, default=1)
    parser.add_argument('--log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")
    parser.add_argument('--create_tf_timeline', type=argparse_bool, default=False,
                        help="Generate tensorflow trace.")

    # Train test split
    parser.add_argument('--train_test_split_ratio', type=float, default=4.0)
    parser.add_argument('--train_test_split_shuffle', type=argparse_bool, default=False)
    parser.add_argument('--max_num_train_files', type=int, default=-1)
    parser.add_argument('--max_num_test_files', type=int, default=-1)
    parser.add_argument('--fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

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
    parser.add_argument('--input_id', type=str, default="in_grid_3d")
    parser.add_argument('--target_id', type=str, default="prob_rewards")
    parser.add_argument('--input_stats_filename', type=str)
    parser.add_argument('--obs_levels_to_use', type=str)
    parser.add_argument('--subvolume_slice_x', type=str)
    parser.add_argument('--subvolume_slice_y', type=str)
    parser.add_argument('--subvolume_slice_z', type=str)
    parser.add_argument('--normalize_input', type=argparse_bool, default=True)
    parser.add_argument('--normalize_target', type=argparse_bool, default=False)

    # # Model parameters
    # parser.add_argument('--num_convs_per_block', type=int, default=2)
    # parser.add_argument('--initial_num_filters', type=int, default=8)
    # parser.add_argument('--filter_increase_per_block', type=int, default=8)
    # parser.add_argument('--filter_increase_within_block', type=int, default=0)
    # parser.add_argument('--maxpool_after_each_block', type=argparse_bool, default=True)
    # parser.add_argument('--max_num_blocks', type=int, default=-1)
    # parser.add_argument('--max_output_grid_size', type=int, default=8)
    # parser.add_argument('--dropout_rate', type=float, default=0.5)
    # parser.add_argument('--add_bias_3dconv', type=argparse_bool, default=False)
    # parser.add_argument('--use_batch_norm_3dconv', type=argparse_bool, default=False)
    # parser.add_argument('--use_batch_norm_regression', type=argparse_bool, default=False)
    # parser.add_argument('--activation_fn_3dconv', type=str, default="relu")
    # parser.add_argument('--num_units_regression', type=str, default="1024")
    # parser.add_argument('--activation_fn_regression', type=str, default="relu")

    args = parser.parse_args()

    run(args)
