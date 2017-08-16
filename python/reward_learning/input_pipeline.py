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


def get_in_grid_3d_normalization(input_stats_filename, filenames=None):
    # Mean and stddev of input for normalization
    if os.path.isfile(input_stats_filename):
        try:
            numpy_dict = data_record.read_hdf5_file_to_numpy_dict(input_stats_filename)
            all_data_size = int(numpy_dict["all_data_size"])
            mean_in_grid_3d_np = numpy_dict["mean_in_grid_3d"]
            stddev_in_grid_3d_np = numpy_dict["stddev_in_grid_3d"]
            mean_z_score = numpy_dict["mean_z_score"]
            stddev_z_score = numpy_dict["stddev_z_score"]
            record_batch = data_record.read_hdf5_records_v3(filenames[0])
            assert(np.all(numpy_dict["mean_in_grid_3d"].shape == record_batch.in_grid_3ds.shape[1:]))
        except Exception, err:
            print("Unable to read data statistics file.")
            raise err
    elif filenames is not None:
        assert(len(filenames) > 0)
        print("Computing data statistics")
        all_data_size, ((mean_in_grid_3d_np, stddev_in_grid_3d_np), (mean_z_score, stddev_z_score)) \
            = data_record.compute_dataset_stats_from_hdf5_files_v3(filenames, ["in_grid_3ds"], compute_z_scores=True)
        if not np.all(np.abs(mean_z_score) < 1e-2):
            print("mean_z_score")
            print(mean_z_score)
            print(mean_z_score[np.abs(mean_z_score) >= 1e-2])
            print(np.sum(np.abs(mean_z_score) >= 1e-2))
            print(np.max(np.abs(mean_z_score)))
        assert(np.all(np.abs(mean_z_score) < 1e-3))
        if np.any(np.abs(stddev_z_score - 1) < 1e-2):
            print("stddev_z_score")
            print(stddev_z_score)
            print(stddev_z_score[np.abs(stddev_z_score - 1) >= 1e-2])
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
    else:
        import exceptions
        raise exceptions.IOError("Data statistics file not found")
    print("Data statistics:")
    print("  Mean of in_grid_3d:", np.mean(mean_in_grid_3d_np.flatten()))
    print("  Stddev of in_grid_3d:", np.mean(stddev_in_grid_3d_np.flatten()))
    print("  Mean of z_score:", np.mean(mean_z_score.flatten()))
    print("  Stddev of z_score:", np.mean(stddev_z_score.flatten()))
    print("  Size of full dataset:", all_data_size)

    return mean_in_grid_3d_np, stddev_in_grid_3d_np


def get_input_from_record_fn(config, subvolume_slices, in_grid_3d_shape,
                             mean_in_grid_3d_np=None, stddev_in_grid_3d_np=None, verbose=False):
    subvolume_slice_x, subvolume_slice_y, subvolume_slice_z = subvolume_slices
    # Retrieval functions for input from data records
    if config.input_id == "in_grid_3d":
        # Determine channels to use
        if config.obs_levels_to_use is None:
            in_grid_3d_channels = range(in_grid_3d_shape[-1])
        else:
            obs_levels_to_use = [int(x) for x in config.obs_levels_to_use.split(',')]
            in_grid_3d_channels = []
            for level in obs_levels_to_use:
                in_grid_3d_channels.append(2 * level)
                in_grid_3d_channels.append(2 * level + 1)
        if verbose:
            print("Channels of in_grid_3d: {}".format(in_grid_3d_channels))

        def get_input_from_record(record):
            return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]

    elif config.input_id.startswith("in_grid_3d["):
        in_channels = config.input_id[len("in_grid_3d"):]
        in_channels = [int(x) for x in in_channels.strip("[]").split(",")]

        def get_input_from_record(record):
            return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_channels]

    elif config.input_id.startswith("rand["):
        shape = config.input_id[len("rand"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_input_from_record(record):
            return np.random.rand(*shape)

    elif config.input_id.startswith("randn["):
        shape = config.input_id[len("randn"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_input_from_record(record):
            return np.random.randn(*shape)

    else:
        raise NotImplementedError("Unknown input name: {}".format(config.input_id))

    if config.normalize_input:
        assert(mean_in_grid_3d_np is not None)
        assert(stddev_in_grid_3d_np is not None)
        mean_record = data_record.RecordV3(None, mean_in_grid_3d_np, None, None, None)
        mean_input_np = get_input_from_record(mean_record)
        stddev_record = data_record.RecordV3(None, stddev_in_grid_3d_np, None, None, None)
        stddev_input_np = get_input_from_record(stddev_record)

        get_unnormalized_input_from_record = get_input_from_record

        # Retrieval functions for normalized input
        def get_input_from_record(record):
            single_input = get_unnormalized_input_from_record(record)
            if config.normalize_input:
                single_input = (single_input - mean_input_np) / stddev_input_np
            return single_input

    return get_input_from_record


def get_target_from_record_fn(config, subvolume_slices,
                              mean_in_grid_3d_np=None, stddev_in_grid_3d_np=None, verbose=False):
    subvolume_slice_x, subvolume_slice_y, subvolume_slice_z = subvolume_slices
    # Retrieval functions for target from data records
    if config.target_id == "reward":
        def get_target_from_record(record):
            return record.rewards[..., 0].reshape((1,))

    elif config.target_id == "norm_reward":
        def get_target_from_record(record):
            return record.rewards[..., 1].reshape((1,))

    elif config.target_id == "prob_reward":
        def get_target_from_record(record):
            return record.rewards[..., 2].reshape((1,))

    elif config.target_id == "norm_prob_reward":
        def get_target_from_record(record):
            return record.rewards[..., 3].reshape((1,))

    elif config.target_id == "score":
        def get_target_from_record(record):
            return record.scores[..., 0].reshape((1,))

    elif config.target_id == "norm_score":
        def get_target_from_record(record):
            return record.scores[..., 1].reshape((1,))

    elif config.target_id == "prob_score":
        def get_target_from_record(record):
            return record.scores[..., 2].reshape((1,))

    elif config.target_id == "norm_prob_score":
        def get_target_from_record(record):
            return record.scores[..., 3].reshape((1,))

    elif config.target_id == "mean_occupancy":
        def get_target_from_record(record):
            return np.mean(record.in_grid_3d[..., 0::2]).reshape((1,))

    elif config.target_id == "sum_occupancy":
        def get_target_from_record(record):
            return np.sum(record.in_grid_3d[..., 0::2]).reshape((1,))

    elif config.target_id == "mean_observation":
        def get_target_from_record(record):
            return np.mean(record.in_grid_3d[..., 1::2]).reshape((1,))

    elif config.target_id == "sum_observation":
        def get_target_from_record(record):
            return np.sum(record.in_grid_3d[..., 1::2]).reshape((1,))

    # elif config.target_id == "in_grid_3d":
    #     assert(config.input_id == "in_grid_3d")
    #
    #     def get_target_from_record(record):
    #         return record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    #
    #
    # elif cfg.data.target_id == "norm_in_grid_3d":
    #     assert(cfg.data.input_id == "in_grid_3d")
    #
    #     def get_target_from_record(record):
    #         single_grid_3d = record.in_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    #         if cfg.data.normalize_input:
    #             single_grid_3d = (single_grid_3d - mean_input_np) / stddev_input_np
    #         return single_grid_3d
    #
    # elif config.target_id == "out_grid_3d":
    #     assert(config.input_id == "in_grid_3d")
    #
    #     def get_target_from_record(record):
    #         return record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    #
    # elif cfg.data.target_id == "norm_out_grid_3d":
    #     assert(cfg.data.input_id == "in_grid_3d")
    #
    #     def get_target_from_record(record):
    #         single_grid_3d = record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, in_grid_3d_channels]
    #         if cfg.data.normalize_input:
    #             single_grid_3d = (single_grid_3d - mean_input_np) / stddev_input_np
    #         return single_grid_3d

    elif config.target_id.startswith("out_grid_3d["):
        out_channels = config.target_id[len("out_grid_3d"):]
        out_channels = [int(x) for x in out_channels.strip("[]").split(",")]

        def get_target_from_record(record):
            return record.out_grid_3d[subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, out_channels]

    elif config.target_id.startswith("rand["):
        shape = config.target_id[len("rand"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_target_from_record(record):
            return np.random.rand(*shape)

    elif config.target_id.startswith("randn["):
        shape = config.target_id[len("randn"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_target_from_record(record):
            return np.random.randn(*shape)

    else:
        raise NotImplementedError("Unknown target name: {}".format(config.target_id))

    if config.normalize_target:
        print("Computing target statistics. This can take a while.")

        def target_batches_generator(filenames):
            for filename in filenames:
                for record in data_record.generate_single_records_from_hdf5_file_v3(filename):
                    single_target = get_target_from_record(record)
                    target_batch = single_target[np.newaxis, ...]
                    yield [target_batch]

        _, ((mean_target_np, stddev_target_np),) \
            = data_record.compute_dataset_stats(target_batches_generator(filenames))
        print("Target data statistics:")
        print("  Mean of target:", np.mean(mean_target_np.flatten()))
        print("  Stddev of target:", np.mean(stddev_target_np.flatten()))

        get_unnormalized_target_from_record = get_target_from_record

        # Retrieval functions for normalized target
        def get_target_from_record(record):
            single_target = get_unnormalized_target_from_record(record)
            if config.normalize_target:
                single_target = (single_target - mean_target_np) / stddev_target_np
            return single_target

    return get_target_from_record


def get_input_and_target_from_record_functions(config, filenames, verbose=False):
    assert(len(filenames) > 0)
    # Input configuration, i.e. which slices and channels from the 3D grids to use.
    # First read any of the data records
    record_batch = data_record.read_hdf5_records_v3(filenames[0])
    assert(record_batch.in_grid_3ds.shape[0] > 0)
    assert(np.all(record_batch.in_grid_3ds.shape == record_batch.out_grid_3ds.shape))
    records = list(data_record.generate_single_records_from_batch_v3(record_batch))
    record = records[0]
    if verbose and config.input_id == "in_grid_3d":
        print("Observation levels in data record: {}".format(record.obs_levels))
    in_grid_3d = record.in_grid_3d

    # Determine subvolume slices
    if config.subvolume_slice_x is None:
        subvolume_slice_x = slice(0, in_grid_3d.shape[0])
    else:
        subvolume_slice_x = slice(*[int(x) for x in config.subvolume_slice_x.split(',')])
    if config.subvolume_slice_y is None:
        subvolume_slice_y = slice(0, in_grid_3d.shape[1])
    else:
        subvolume_slice_y = slice(*[int(x) for x in config.subvolume_slice_y.split(',')])
    if config.subvolume_slice_z is None:
        subvolume_slice_z = slice(0, in_grid_3d.shape[2])
    else:
        subvolume_slice_z = slice(*[int(x) for x in config.subvolume_slice_z.split(',')])
    if verbose:
        # Print used subvolume slices and channels
        print("Subvolume slice x: {}".format(subvolume_slice_x))
        print("subvolume slice y: {}".format(subvolume_slice_y))
        print("subvolume slice z: {}".format(subvolume_slice_z))
        print("Input id: {}".format(config.input_id))
        print("Target id: {}".format(config.target_id))

    if config.normalize_input:
        input_stats_filename = config.input_stats_filename
        mean_in_grid_3d_np, stddev_in_grid_3d_np = get_in_grid_3d_normalization(input_stats_filename, filenames)
    else:
        mean_in_grid_3d_np = None
        stddev_in_grid_3d_np = None

    subvolume_slices = [subvolume_slice_x, subvolume_slice_y, subvolume_slice_z]
    get_input_from_record = get_input_from_record_fn(
        config, subvolume_slices, in_grid_3d.shape,
        mean_in_grid_3d_np, stddev_in_grid_3d_np, verbose=verbose)

    if config.target_id == "input":
        def get_target_from_record(record):
            return get_input_from_record(record)
    else:
        get_target_from_record = get_target_from_record_fn(
            config, subvolume_slices,
            mean_in_grid_3d_np, stddev_in_grid_3d_np, verbose=verbose)

    # Retrieve input and target shapes
    in_grid_3d = record.in_grid_3d
    in_grid_3d_shape = list(in_grid_3d.shape)
    input_shape = list(get_input_from_record(record).shape)
    target_shape = list(get_target_from_record(record).shape)
    if verbose:
        print("Input and target shapes:")
        print("  Shape of grid_3d: {}".format(in_grid_3d_shape))
        print("  Shape of input: {}".format(input_shape))
        print("  Shape of target: {}".format(target_shape))

    return input_shape, get_input_from_record, target_shape, get_target_from_record


def print_data_stats(filename, get_input_from_record, get_target_from_record):
    # Input configuration, i.e. which slices and channels from the 3D grids to use.
    # First read any of the data records
    record_batch = data_record.read_hdf5_records_v3(filename)
    assert(record_batch.in_grid_3ds.shape[0] > 0)
    assert(np.all(record_batch.in_grid_3ds.shape == record_batch.out_grid_3ds.shape))
    records = list(data_record.generate_single_records_from_batch_v3(record_batch))

    # Report some stats on input and outputs for the first data file
    # This is only for sanity checking
    # TODO: This can be confusing as we just take mean and average over all 3d positions
    print("Stats on inputs and outputs for single file")
    inputs = [get_input_from_record(record) for record in records]
    for i in xrange(inputs[0].shape[-1]):
        values = [input[..., i] for input in inputs]
        print("  Mean of input {}: {}".format(i, np.mean(values)))
        print("  Stddev of input {}: {}".format(i, np.std(values)))
        print("  Min of input {}: {}".format(i, np.min(values)))
        print("  Max of input {}: {}".format(i, np.max(values)))
    targets = [get_target_from_record(record) for record in records]
    if len(targets[0].shape) > 1:
        for i in xrange(targets[0].shape[-1]):
            values = [target[..., i] for target in targets]
            print("  Mean of target {}: {}".format(i, np.mean(values)))
            print("  Stddev of target {}: {}".format(i, np.std(values)))
            print("  Min of target {}: {}".format(i, np.min(values)))
            print("  Max of target {}: {}".format(i, np.max(values)))
    values = [target for target in targets]
    print("  Mean of target: {}".format(np.mean(values)))
    print("  Stddev of target: {}".format(np.std(values)))
    print("  Min of target: {}".format(np.min(values)))
    print("  Max of target: {}".format(np.max(values)))

    # Retrieve input and target shapes
    in_grid_3d = record.in_grid_3d
    in_grid_3d_shape = list(in_grid_3d.shape)
    input_shape = list(get_input_from_record(record).shape)
    target_shape = list(get_target_from_record(record).shape)
    print("Input and target shapes:")
    print("  Shape of grid_3d: {}".format(in_grid_3d_shape))
    print("  Shape of input: {}".format(input_shape))
    print("  Shape of target: {}".format(target_shape))


class InputPipeline(object):

    def __init__(self, sess, coord, filenames, parse_record_fn,
                 input_shape, target_shape, batch_size,
                 queue_capacity, min_after_dequeue,
                 shuffle, num_threads, timeout=60,
                 fake_constant_data=False, fake_random_data=False,
                 name="", verbose=False,):
        self._parse_record_fn = parse_record_fn
        self._input_shape = input_shape
        self._target_shape = target_shape
        self._fake_constant_data = fake_constant_data
        self._fake_random_data = fake_random_data

        # Create HDF5 readers
        self._hdf5_input_pipeline = data_record.HDF5ReaderProcessCoordinator(
            filenames, coord, shuffle=shuffle, hdf5_record_version=data_record.HDF5_RECORD_VERSION_3,
            timeout=timeout, num_processes=num_threads, verbose=verbose)
        self._num_records = None

        tensor_dtypes = [tf.float32, tf.float32]
        tensor_shapes = [input_shape, target_shape]
        self._tf_pipeline = data_provider.TFInputPipeline(
            self._input_and_target_provider_factory(self._hdf5_input_pipeline),
            sess, coord, batch_size, tensor_shapes, tensor_dtypes,
            queue_capacity=queue_capacity,
            min_after_dequeue=min_after_dequeue,
            shuffle=shuffle,
            num_threads=num_threads,
            timeout=timeout,
            name="{}_tf_input_pipeline".format(name),
            verbose=verbose)

        # Retrieve tensors from data bridge
        self._input_batch, self._target_batch = self._tf_pipeline.tensors

    def _input_and_target_provider_factory(self, hdf5_input_pipeline):
        def input_and_target_provider():
            if self._fake_constant_data:
                single_input = np.ones(self._input_shape)
                single_target = np.ones(self._target_shape)
            elif self._fake_random_data:
                single_input = np.random.randn(self._input_shape)
                single_target = np.random.randn(self._target_shape)
            else:
                record = hdf5_input_pipeline.get_next_record()
                single_input, single_target = self._parse_record_fn(record)
            assert(np.all(np.isfinite(single_input)))
            assert(np.all(np.isfinite(single_target)))
            return single_input, single_target
        return input_and_target_provider

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
