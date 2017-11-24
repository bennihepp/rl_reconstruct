#!/usr/bin/env python

from __future__ import print_function

import data_record
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorpack.dataflow
import data_provider
from pybh import hdf5_utils
from pybh import tensorpack_utils
from pybh import log_utils


logger = log_utils.get_logger("reward_learning/input_pipeline")


def compute_data_statistics_from_hdf5_files(data_filenames):
    assert(len(data_filenames) > 0)
    logger.info("Computing data statistics")
    field_names = ["in_grid_3ds", "out_grid_3ds", "rewards", "scores", "rgb_images", "depth_images", "normal_images"]
    statistics_dict, z_score_statistics_dict = data_record.compute_dataset_stats_from_hdf5_files_v4(
        data_filenames, field_names, compute_z_scores=True)
    logger.info("Data statistics:")
    logger.info("  Mean of in_grid_3d:", np.mean(statistics_dict["in_grid_3ds"]["mean"].flatten()))
    logger.info("  Stddev of in_grid_3d:", np.mean(statistics_dict["in_grid_3ds"]["stddev"].flatten()))
    logger.info("  Mean of z_score:", np.mean(z_score_statistics_dict["in_grid_3ds"]["mean"].flatten()))
    logger.info("  Stddev of z_score:", np.mean(z_score_statistics_dict["in_grid_3ds"]["stddev"].flatten()))
    logger.info("  Size of dataset:", statistics_dict["in_grid_3ds"]["num_samples"])
    mean_z_score_tolerance = 1e-3
    stddev_z_score_tolerance = 1e-2
    for key in z_score_statistics_dict:
        z_score_statistics = z_score_statistics_dict[key]
        if not np.all(np.abs(z_score_statistics["mean"]) < mean_z_score_tolerance):
            logger.info("mean z-score for field {}".format(key))
            logger.info(z_score_statistics_dict["mean"])
            logger.info(z_score_statistics_dict["mean"][np.abs(z_score_statistics_dict["mean"]) >= mean_z_score_tolerance])
            logger.info(np.sum(np.abs(z_score_statistics["mean"]) >= mean_z_score_tolerance))
            logger.info(np.max(np.abs(z_score_statistics["mean"])))
        assert(np.all(np.abs(z_score_statistics["mean"]) < mean_z_score_tolerance))
        if not np.all(np.abs(z_score_statistics["stddev"] - 1) < stddev_z_score_tolerance):
            logger.info("stddev z-score for field {}".format(key))
            logger.info(z_score_statistics["stddev"])
            logger.info(z_score_statistics["stddev"][np.abs(z_score_statistics["stddev"] - 1) >= stddev_z_score_tolerance])
            logger.info(np.sum(np.abs(z_score_statistics["stddev"] - 1) >= stddev_z_score_tolerance))
            logger.info(np.max(np.abs(z_score_statistics["stddev"] - 1)))
        assert(np.all(np.abs(z_score_statistics["stddev"] - 1) < stddev_z_score_tolerance))
    statistics_dict["_z_score_"] = z_score_statistics_dict
    return statistics_dict


def compute_data_statistics_from_hdf5_files(data_generator):
    logger.info("Computing data statistics")
    field_names = ["in_grid_3ds", "out_grid_3ds", "rewards", "scores", "rgb_images", "depth_images", "normal_images"]
    statistics_dict, z_score_statistics_dict = data_record.compute_dataset_stats(data_generator, field_names, compute_z_scores=True)
    logger.info("Data statistics:")
    logger.info("  Mean of in_grid_3d:", np.mean(statistics_dict["in_grid_3ds"]["mean"].flatten()))
    logger.info("  Stddev of in_grid_3d:", np.mean(statistics_dict["in_grid_3ds"]["stddev"].flatten()))
    logger.info("  Mean of z_score:", np.mean(z_score_statistics_dict["in_grid_3ds"]["mean"].flatten()))
    logger.info("  Stddev of z_score:", np.mean(z_score_statistics_dict["in_grid_3ds"]["stddev"].flatten()))
    logger.info("  Size of dataset:", statistics_dict["in_grid_3ds"]["num_samples"])
    mean_z_score_tolerance = 1e-3
    stddev_z_score_tolerance = 1e-2
    for key in z_score_statistics_dict:
        z_score_statistics = z_score_statistics_dict[key]
        if not np.all(np.abs(z_score_statistics["mean"]) < mean_z_score_tolerance):
            logger.info("mean z-score for field {}".format(key))
            logger.info(z_score_statistics_dict["mean"])
            logger.info(z_score_statistics_dict["mean"][np.abs(z_score_statistics_dict["mean"]) >= mean_z_score_tolerance])
            logger.info(np.sum(np.abs(z_score_statistics["mean"]) >= mean_z_score_tolerance))
            logger.info(np.max(np.abs(z_score_statistics["mean"])))
        assert(np.all(np.abs(z_score_statistics["mean"]) < mean_z_score_tolerance))
        if not np.all(np.abs(z_score_statistics["stddev"] - 1) < stddev_z_score_tolerance):
            logger.info("stddev z-score for field {}".format(key))
            logger.info(z_score_statistics["stddev"])
            logger.info(z_score_statistics["stddev"][np.abs(z_score_statistics["stddev"] - 1) >= stddev_z_score_tolerance])
            logger.info(np.sum(np.abs(z_score_statistics["stddev"] - 1) >= stddev_z_score_tolerance))
            logger.info(np.max(np.abs(z_score_statistics["stddev"] - 1)))
        assert(np.all(np.abs(z_score_statistics["stddev"] - 1) < stddev_z_score_tolerance))
    statistics_dict["_z_score_"] = z_score_statistics_dict
    return statistics_dict


def count_data_samples(data_filenames):
    assert(len(data_filenames) > 0)
    logger.info("Counting data samples")

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    try:
        sample_counts_async = p.map_async(data_record.count_records_in_hdf5_file_v4, data_filenames, chunksize=4)
        # Workaround to enable KeyboardInterrupt (with p.map() or record_counts_async.get() it won't be received)
        sample_counts = sample_counts_async.get(timeout=np.iinfo(np.int32).max)
        sample_count = np.sum(sample_counts)
        p.close()
        p.join()
    except Exception as exc:
        logger.info("Exception occured when computing num records: {}".format(exc))
        p.close()
        p.terminate()
        raise
    return sample_count


def read_data_statistics(stats_filename):
    statistics_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(stats_filename)
    return statistics_dict


def write_data_statistics(stats_filename, statistics_dict):
    hdf5_utils.write_numpy_dict_to_hdf5_file(stats_filename, statistics_dict)


def get_samples_from_data_fn(config, data_stats, batch_data, verbose=False):
    # Which slices and channels from the 3D grids to use.

    # Determine subvolume slices
    if "subvolume_slice_x" in config:
        subvolume_slice_x = slice(*[int(x) for x in config.subvolume_slice_x.split(':')])
    else:
        subvolume_slice_x = slice(None)
    if "subvolume_slice_y" in config:
        subvolume_slice_y = slice(*[int(x) for x in config.subvolume_slice_y.split(':')])
    else:
        subvolume_slice_y = slice(None)
    if "subvolume_slice_z" in config:
        subvolume_slice_z = slice(*[int(x) for x in config.subvolume_slice_z.split(':')])
    else:
        subvolume_slice_z = slice(None)
    if verbose:
        # Print used subvolume slices and channels
        logger.info("  Subvolume slice x: {}".format(subvolume_slice_x))
        logger.info("  subvolume slice y: {}".format(subvolume_slice_y))
        logger.info("  subvolume slice z: {}".format(subvolume_slice_z))

    normalize = config.get("normalize", False)
    normalize_per_element = config.get("normalize_per_element", True)
    mean_sample_np = None
    stddev_sample_np = None
    num_samples = None
    sample_field_name = None

    # Retrieval functions for samples from data
    if config.id == "in_grid_3d":
        # Determine channels to use
        if config.get("obs_levels_to_use", None) is None:
            channels = slice(None)
        elif config.obs_levels_to_use.find(":") >= 0:
            channels = slice(*[int(x) for x in config.obs_levels_to_use.split(':')])
        else:
            obs_levels_to_use = [int(x) for x in config.obs_levels_to_use.split(',')]
            channels = []
            for level in obs_levels_to_use:
                channels.append(2 * level)
                channels.append(2 * level + 1)
        if verbose:
            logger.info("Channels of in_grid_3d: {}".format(channels))

        if normalize:
            mean_sample_np = data_stats["in_grid_3ds"]["mean"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            stddev_sample_np = data_stats["in_grid_3ds"]["stddev"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        def get_samples_from_data(data):
            return data[sample_field_name][..., subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]

    elif config.id.startswith("in_grid_3d["):
        channels = config.id[len("in_grid_3d"):].strip("[]")
        logger.info("channels: {}".format(channels))
        if channels.find(":") >= 0:
            channels = slice(*[int(x) for x in channels.split(':')])
        else:
            channels = [int(x) for x in channels.split(",")]

        if verbose:
            logger.info("Channels of in_grid_3d: {}".format(channels))

        if normalize:
            mean_sample_np = data_stats["in_grid_3ds"]["mean"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            stddev_sample_np = data_stats["in_grid_3ds"]["stddev"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        def get_samples_from_data(data):
            return data[sample_field_name][..., subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]

    elif config.id == "out_grid_3d":
        # Determine channels to use
        if config.get("obs_levels_to_use", None) is None:
            channels = slice(None)
        elif config.obs_levels_to_use.find(":") >= 0:
            channels = slice(*[int(x) for x in config.obs_levels_to_use.split(':')])
        else:
            obs_levels_to_use = [int(x) for x in config.obs_levels_to_use.split(',')]
            channels = []
            for level in obs_levels_to_use:
                channels.append(2 * level)
                channels.append(2 * level + 1)
        if verbose:
            logger.info("Channels of out_grid_3d: {}".format(channels))

        if normalize:
            mean_sample_np = data_stats["out_grid_3ds"]["mean"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            stddev_sample_np = data_stats["out_grid_3ds"]["stddev"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            num_samples = data_stats["out_grid_3ds"]["num_samples"]
        sample_field_name = "out_grid_3ds"

        def get_samples_from_data(data):
            return data[sample_field_name][..., subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]

    elif config.id.startswith("out_grid_3d["):
        channels = config.id[len("out_grid_3d"):].strip("[]")
        if channels.find(":") >= 0:
            channels = slice(*[int(x) for x in channels.split(':')])
        else:
            channels = [int(x) for x in channels.split(",")]

        if normalize:
            mean_sample_np = data_stats["out_grid_3ds"]["mean"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            stddev_sample_np = data_stats["out_grid_3ds"]["stddev"][subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]
            num_samples = data_stats["out_grid_3ds"]["num_samples"]
        sample_field_name = "out_grid_3ds"

        def get_samples_from_data(data):
            return data[sample_field_name][..., subvolume_slice_x, subvolume_slice_y, subvolume_slice_z, channels]

    elif config.id == "depth_image":
        if normalize:
            mean_sample_np = data_stats["depth_images"]["mean"][..., np.newaxis]
            stddev_sample_np = data_stats["depth_images"]["stddev"][..., np.newaxis]
            num_samples = data_stats["depth_images"]["num_samples"]
        sample_field_name = "depth_images"
        if "image_size" in config:
            import cv2
            image_size = tuple(config.image_size)

            if normalize:
                mean_sample_np = cv2.resize(mean_sample_np, image_size, interpolation=cv2.INTER_CUBIC)
                stddev_sample_np = cv2.resize(stddev_sample_np, image_size, interpolation=cv2.INTER_CUBIC)

            if batch_data:
                def get_samples_from_data(data):
                    depth_images = data[sample_field_name][..., np.newaxis]
                    resized_depth_images = np.empty((depth_images.shape[0], image_size[0], image_size[1], 1))
                    for i in depth_images.shape[0]:
                        resized_depth_images[i, ...] = cv2.resize(depth_images[i, ...], image_size,
                                                                  interpolation=cv2.INTER_CUBIC)
                    return resized_depth_images
            else:
                def get_samples_from_data(data):
                    depth_image = data[sample_field_name][..., np.newaxis]
                    resized_depth_image = cv2.resize(depth_image, image_size, interpolation=cv2.INTER_CUBIC)
                    return resized_depth_image

        else:
            def get_samples_from_data(data):
                return data[sample_field_name][..., np.newaxis]

    elif config.id == "reward":
        if normalize:
            mean_sample_np = data_stats["rewards"]["mean"][0:1]
            stddev_sample_np = data_stats["rewards"]["stddev"][0:1]
            num_samples = data_stats["rewards"]["num_samples"]
        sample_field_name = "rewards"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 0:1]

    elif config.id == "norm_reward":
        if normalize:
            mean_sample_np = data_stats["rewards"]["mean"][1:2]
            stddev_sample_np = data_stats["rewards"]["stddev"][1:2]
            num_samples = data_stats["rewards"]["num_samples"]
        sample_field_name = "rewards"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 1:2]

    elif config.id == "prob_reward":
        if normalize:
            mean_sample_np = data_stats["rewards"]["mean"][2:3]
            stddev_sample_np = data_stats["rewards"]["stddev"][2:3]
            num_samples = data_stats["rewards"]["num_samples"]
        sample_field_name = "rewards"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 2:3]

    elif config.id == "norm_prob_reward":
        if normalize:
            mean_sample_np = data_stats["rewards"]["mean"][3:4]
            stddev_sample_np = data_stats["rewards"]["stddev"][3:4]
            num_samples = data_stats["rewards"]["num_samples"]
        sample_field_name = "rewards"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 3:4]

    elif config.id == "score":
        if normalize:
            mean_sample_np = data_stats["scores"]["mean"][0:1]
            stddev_sample_np = data_stats["scores"]["stddev"][0:1]
            num_samples = data_stats["scores"]["num_samples"]
        sample_field_name = "scores"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 0:1]

    elif config.id == "norm_score":
        if normalize:
            mean_sample_np = data_stats["scores"]["mean"][1:2]
            stddev_sample_np = data_stats["scores"]["stddev"][1:2]
            num_samples = data_stats["scores"]["num_samples"]
        sample_field_name = "scores"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 1:2]

    elif config.id == "prob_score":
        if normalize:
            mean_sample_np = data_stats["scores"]["mean"][2:3]
            stddev_sample_np = data_stats["scores"]["stddev"][2:3]
            num_samples = data_stats["scores"]["num_samples"]
        sample_field_name = "scores"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 2:3]

    elif config.id == "norm_prob_score":
        if normalize:
            mean_sample_np = data_stats["scores"]["mean"][3:4]
            stddev_sample_np = data_stats["scores"]["stddev"][3:4]
            num_samples = data_stats["scores"]["num_samples"]
        sample_field_name = "scores"

        def get_samples_from_data(data):
            return data[sample_field_name][..., 3:4]

    elif config.id == "mean_occupancy":
        if normalize:
            mean_sample_np = np.mean(data_stats["in_grid_3ds"]["mean"][..., 0::2])
            stddev_sample_np = np.mean(data_stats["in_grid_3ds"]["stddev"][..., 0::2])
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        if batch_data:
            def get_samples_from_data(data):
                mean_occupancy = np.empty((data[sample_field_name].shape[0]))
                for i in range(data[sample_field_name].shape[0]):
                    mean_occupancy[i] = np.mean(data[sample_field_name][i, ..., 0::2])
                return mean_occupancy
        else:
            def get_samples_from_data(data):
                return np.mean(data[sample_field_name][..., 0::2])

    elif config.id == "sum_occupancy":
        if normalize:
            mean_sample_np = np.sum(data_stats["in_grid_3ds"]["mean"][..., 0::2])
            stddev_sample_np = np.sum(data_stats["in_grid_3ds"]["stddev"][..., 0::2])
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        if batch_data:
            def get_samples_from_data(data):
                sum_occupancy = np.empty((data[sample_field_name].shape[0]))
                for i in range(data[sample_field_name].shape[0]):
                    sum_occupancy[i] = np.sum(data[sample_field_name][i, ..., 0::2])
                return sum_occupancy
        else:
            def get_samples_from_data(data):
                return np.sum(data[sample_field_name][..., 0::2])

    elif config.id == "mean_observation":
        if normalize:
            mean_sample_np = np.mean(data_stats["in_grid_3ds"]["mean"][..., 1::2])
            stddev_sample_np = np.mean(data_stats["in_grid_3ds"]["stddev"][..., 1::2])
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        if batch_data:
            def get_samples_from_data(data):
                mean_observation = np.empty((data[sample_field_name].shape[0]))
                for i in range(data[sample_field_name].shape[0]):
                    mean_observation[i] = np.mean(data[sample_field_name][i, ..., 1::2])
                return mean_observation
        else:
            def get_samples_from_data(data):
                return np.mean(data[sample_field_name][..., 1::2])

    elif config.id == "sum_observation":
        if normalize:
            mean_sample_np = np.sum(data_stats["in_grid_3ds"]["mean"][..., 1::2])
            stddev_sample_np = np.sum(data_stats["in_grid_3ds"]["stddev"][..., 1::2])
            num_samples = data_stats["in_grid_3ds"]["num_samples"]
        sample_field_name = "in_grid_3ds"

        if batch_data:
            def get_samples_from_data(data):
                sum_observation = np.empty((data[sample_field_name].shape[0]))
                for i in range(data[sample_field_name].shape[0]):
                    sum_observation[i] = np.sum(data[sample_field_name][i, ..., 1::2])
                return sum_observation
        else:
            def get_samples_from_data(data):
                return np.sum(data[sample_field_name][..., 1::2])

    elif config.id.startswith("rand["):
        assert(not normalize)
        shape = config.id[len("rand"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_samples_from_data(_):
            return np.random.rand(*shape)

    elif config.id.startswith("randn["):
        assert(not normalize)
        shape = config.id[len("randn"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_samples_from_data(_):
            return np.random.randn(*shape)

    elif config.id.startswith("ones["):
        assert(not normalize)
        shape = config.id[len("ones"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_samples_from_data(_):
            return np.ones(shape)

    elif config.id.startswith("zeros["):
        assert(not normalize)
        shape = config.id[len("zeros"):]
        shape = [int(x) for x in shape.strip("[]").split(",")]

        def get_samples_from_data(_):
            return np.zeros(shape)

    else:
        raise NotImplementedError("Unknown sample id: {}".format(config.id))

    # TODO: This should be fixed in the dataset
    if "clip_min" in config:
        logger.warn("-------------------------------------------------------------------")
        logger.warn("WARNING: Clipping sample values should be performed in the dataset!")
        logger.warn("-------------------------------------------------------------------")
        clip_min = float(config.clip_min)
        get_samples_from_data_no_min_clip = get_samples_from_data

        def get_samples_from_record(data):
            samples = get_samples_from_data_no_min_clip(data)
            samples[samples < clip_min] = clip_min
            return samples

    if "clip_max" in config:
        logger.warn("-------------------------------------------------------------------")
        logger.warn("WARNING: Clipping sample values should be performed in the dataset!")
        logger.warn("-------------------------------------------------------------------")
        clip_max = float(config.clip_max)
        get_samples_from_data_no_max_clip = get_samples_from_data

        def get_samples_from_data(data):
            samples = get_samples_from_data_no_max_clip(data)
            samples[samples > clip_max] = clip_max
            return samples

    if normalize:
        if mean_sample_np is None:
            raise NotImplementedError("Normalized samples are not supported for id {}".format(config.id))
        assert stddev_sample_np is not None
        assert num_samples is not None

        if not normalize_per_element:
            num_elements = np.prod(mean_sample_np.shape[:-1])
            reduced_mean_sample = np.zeros((mean_sample_np.shape[-1],))
            reduced_stddev_sample = np.zeros((mean_sample_np.shape[-1],))
            squared_deviations = np.zeros((mean_sample_np.shape[-1],))
            # logger.info(stddev_sample_np.shape[-1])
            # logger.info(data_stats["in_grid_3ds"]["mean"])
            # logger.info(data_stats["in_grid_3ds"]["stddev"])
            for c in range(stddev_sample_np.shape[-1]):
                # logger.info(c)
                reduced_mean_sample[c] = np.mean(mean_sample_np[..., c])
                squared_deviations[c] = np.sum((num_samples - 1) * (stddev_sample_np[..., c] ** 2))
                squared_deviations[c] = squared_deviations[c] + np.sum(num_samples * (mean_sample_np[..., c] - reduced_mean_sample[c]))
                reduced_stddev_sample[c] = np.sqrt(squared_deviations[c] / (num_samples * num_elements - 1))
                mean_sample_np[..., c] = reduced_mean_sample[c]
                stddev_sample_np[..., c] = reduced_stddev_sample[c]
            logger.info("Reduced mean sample: {}".format(reduced_mean_sample))
            logger.info("Reduced stddev sample: {}".format(reduced_stddev_sample))
            # logger.info("Reduced mean sample: {}".format(mean_sample_np))
            # logger.info("Reduced stddev sample: {}".format(stddev_sample_np))
        else:
            logger.warn("Normalizing per element.")

        # Retrieval functions for normalized samples
        if batch_data:
            mean_sample_batch_np = mean_sample_np[np.newaxis, ...]
            stddev_sample_batch_np = stddev_sample_np[np.newaxis, ...]

            def normalize_samples(samples):
                samples = (samples - mean_sample_batch_np) / stddev_sample_batch_np
                return samples
        else:
            def normalize_samples(samples):
                samples = (samples - mean_sample_np) / stddev_sample_np
                return samples
    else:
        def normalize_samples(samples):
            return samples

    return get_samples_from_data, normalize_samples, mean_sample_np, stddev_sample_np, sample_field_name


class InputAndTargetFromData(object):

    def __init__(self, config, data_stats, batch_data=True, verbose=False):
        self._get_input_from_data_function(config, data_stats, batch_data, verbose)
        self._get_target_from_data_function(config, data_stats, batch_data, verbose)

    def _get_input_from_data_function(self, config, data_stats, batch_data, verbose=False):
        self._input_from_data_fn, self._normalize_input_fn,\
        self._input_mean, self._input_stddev, self._input_field_name \
            = get_samples_from_data_fn(config.input, data_stats, batch_data, verbose=verbose)
        # Kind of hack
        if self._input_mean is not None:
            self._input_mean = np.asarray(self._input_mean, dtype=np.float32)
            self._input_stddev = np.asarray(self._input_stddev, dtype=np.float32)

    def _get_target_from_data_function(self, config, data_stats, batch_data, verbose=False):
        if config.target.id == "input":
            self._target_from_data_fn = self._input_from_data_fn
            self._target_mean = self._input_mean
            self._target_stddev = self._input_stddev
        else:
            self._target_from_data_fn, self._normalize_target_fn, \
            self._target_mean, self._target_stddev, self._target_field_name = \
                get_samples_from_data_fn(config.target, data_stats, batch_data, verbose=verbose)
            if self._target_mean is not None:
                self._target_mean = np.asarray(self._target_mean, dtype=np.float32)
                self._target_stddev = np.asarray(self._target_stddev, dtype=np.float32)

    def get_input_from_data(self, data):
        return self._input_from_data_fn(data)

    def get_target_from_data(self, data):
        return self._target_from_data_fn(data)

    def get_normalized_input_from_data(self, data):
        input_ = self.get_input_from_data(data)
        norm_input = self._normalize_input_fn(input_)
        return norm_input

    def get_normalized_target_from_data(self, data):
        target = self.get_target_from_data(data)
        norm_target = self._normalize_target_fn(target)
        return norm_target

    def normalize_input(self, input_):
        return self._normalize_input_fn(input_)

    def normalize_target(self, target):
        return self._normalize_target_fn(target)

    def denormalize_input(self, input_):
        if self._input_mean is None:
            return input_
        else:
            return (input_ * self._input_stddev) + self._input_mean

    def denormalize_target(self, target):
        if self._target_mean is None:
            return target
        else:
            return (target * self._target_stddev) + self._target_mean

    def get_normalized_input_from_data(self, data):
        return self.normalize_input(self.get_input_from_data(data))

    def get_normalized_target_from_data(self, data):
        return self.normalize_target(self.get_target_from_data(data))

    @property
    def input_field_name(self):
        return self._input_field_name

    @property
    def target_field_name(self):
        return self._target_field_name

    @property
    def input_mean(self):
        return self._input_mean

    @property
    def input_stddev(self):
        return self._input_stddev

    @property
    def target_mean(self):
        return self._target_mean

    @property
    def target_stddev(self):
        return self._target_stddev

    @property
    def input_stats(self):
        return {"mean": self._input_mean, "stddev": self._input_stddev}

    @property
    def target_stats(self):
        return {"mean": self._target_mean, "stddev": self._target_stddev}


class InputAndTargetFromHDF5(InputAndTargetFromData):

    def read_data_from_file(self, filename):
        field_dict = {self._input_field_name: False, self._target_field_name: False}
        data = hdf5_utils.read_hdf5_file_to_numpy_dict(filename, field_dict)
        if self._input_field_name is not None:
            assert(field_dict[self._input_field_name])
        if self._target_field_name is not None:
            assert(field_dict[self._target_field_name])
        return data


def print_data_stats(filename, input_and_target_retriever):
    # Input configuration, i.e. which slices and channels from the 3D grids to use.
    # First read any of the data records
    data = input_and_target_retriever.read_data_from_file(filename)

    # Report some stats on input and outputs for the first data file
    # This is only for sanity checking
    # TODO: This can be confusing as we just take mean and average over all 3d positions
    logger.info("Stats on inputs and outputs for single file")
    input_batch = input_and_target_retriever.get_input_from_data(data)
    for i in range(input_batch.shape[-1]):
        values = input_batch[..., i]
        logger.info("  Mean of input {}: {}".format(i, np.mean(values)))
        logger.info("  Stddev of input {}: {}".format(i, np.std(values)))
        logger.info("  Min of input {}: {}".format(i, np.min(values)))
        logger.info("  Max of input {}: {}".format(i, np.max(values)))
    target_batch = input_and_target_retriever.get_target_from_data(data)
    if len(target_batch.shape) > 2:
        for i in range(target_batch.shape[-1]):
            values = target_batch[..., i]
            logger.info("  Mean of target {}: {}".format(i, np.mean(values)))
            logger.info("  Stddev of target {}: {}".format(i, np.std(values)))
            logger.info("  Min of target {}: {}".format(i, np.min(values)))
            logger.info("  Max of target {}: {}".format(i, np.max(values)))
    values = target_batch
    logger.info("  Mean of target: {}".format(np.mean(values)))
    logger.info("  Stddev of target: {}".format(np.std(values)))
    logger.info("  Min of target: {}".format(np.min(values)))
    logger.info("  Max of target: {}".format(np.max(values)))

    # Retrieve input and target shapes
    in_grid_3d = data["in_grid_3ds"]
    in_grid_3d_shape = list(in_grid_3d.shape)
    input_shape = input_batch.shape
    target_shape = target_batch.shape
    logger.info("Input and target shapes:")
    logger.info("  Shape of grid_3d: {}".format(in_grid_3d_shape))
    logger.info("  Shape of input: {}".format(input_shape))
    logger.info("  Shape of target: {}".format(target_shape))


class InputPipeline(object):

    def _read_data_fn(self, filename):
        data = self._input_and_target_retriever.read_data_from_file(filename)
        return data

    def __init__(self, sess, coord, filenames, input_and_target_retriever,
                 input_shape, target_shape, batch_size, num_samples,
                 data_queue_capacity, sample_queue_capacity, min_after_dequeue,
                 shuffle, num_threads, num_processes, repeats=-1, timeout=60,
                 fake_constant_data=False, fake_random_data=False,
                 name="", verbose=False,
                 tf_dtype=tf.float32):
        self._input_and_target_retriever = input_and_target_retriever
        self._input_shape = input_shape
        self._target_shape = target_shape
        self._fake_constant_data = fake_constant_data
        self._fake_random_data = fake_random_data
        self._num_samples = num_samples

        # Create HDF5 readers
        self._hdf5_input_pipeline = hdf5_utils.HDF5ReaderProcessCoordinator(
            filenames, coord, read_data_fn=self._read_data_fn,
            shuffle=shuffle, repeats=repeats, timeout=timeout, num_processes=num_processes,
            data_queue_capacity=data_queue_capacity, verbose=verbose)

        weights_shape = []
        tensor_dtypes = [tf_dtype, tf_dtype, tf_dtype]
        tensor_shapes = [input_shape, target_shape, weights_shape]
        self._tf_input_pipeline = data_provider.TFInputPipeline(
            self._input_and_target_batch_provider_factory(self._hdf5_input_pipeline, batch_size),
            sess, coord, batch_size, tensor_shapes, tensor_dtypes,
            provides_batches=True,
            queue_capacity=sample_queue_capacity,
            min_after_dequeue=min_after_dequeue,
            shuffle=shuffle,
            num_threads=num_threads,
            timeout=timeout,
            name="{}_tf_input_pipeline".format(name),
            verbose=verbose)

        # Retrieve tensors from data bridge
        self._normalized_input_batch, self._normalized_target_batch, self._weights_batch \
            = self._tf_input_pipeline.tensors_batch

        self._input_batch = self._input_and_target_retriever.unnormalize_input_tensor(self._normalized_input_batch)
        self._target_batch = self._input_and_target_retriever.unnormalize_target_tensor(self._normalized_target_batch)

        # self._timer = Timer()

    def _input_and_target_batch_provider_factory(self, hdf5_input_pipeline, batch_size):
        if self._fake_constant_data:
            def input_and_target_batch_provider():
                input_batch = np.ones((batch_size,) + self._input_shape)
                target_batch = np.ones((batch_size,) + self._target_shape)
                weights_batch = np.ones([input_batch.shape[0]])
                return input_batch, target_batch, weights_batch
        elif self._fake_random_data:
            def input_and_target_batch_provider():
                input_batch = np.random.randn(batch_size, *self._input_shape)
                target_batch = np.random.randn(batch_size, *self._target_shape)
                weights_batch = np.ones([input_batch.shape[0]])
                return input_batch, target_batch, weights_batch
        else:
            def input_and_target_batch_provider():
                data = hdf5_input_pipeline.get_next_data()
                input_batch = self._input_and_target_retriever.get_input_from_data(data)
                target_batch = self._input_and_target_retriever.get_target_from_data(data)
                weights_batch = np.ones([input_batch.shape[0]])
                return input_batch, target_batch, weights_batch

        return input_and_target_batch_provider

    def stop(self):
        self._hdf5_input_pipeline.stop()

    def start(self):
        self._hdf5_input_pipeline.start()
        self._tf_input_pipeline.start()

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def hdf5_input_pipeline(self):
        return self._hdf5_input_pipeline

    @property
    def tf_input_pipeline(self):
        return self._tf_input_pipeline

    @property
    def normalized_tensors_batch(self):
        return self._tf_input_pipeline.tensors_batch

    @property
    def weights_batch(self):
        return self._weights_batch

    @property
    def unnormalized_input_batch(self):
        return self._input_batch

    @property
    def normalized_input_batch(self):
        return self._normalized_input_batch

    @property
    def unnormalized_target_batch(self):
        return self._target_batch

    @property
    def normalized_target_batch(self):
        return self._normalized_target_batch

    @property
    def threads(self):
        return [self._hdf5_input_pipeline.filename_push_thread, self._hdf5_input_pipeline.data_pull_thread] \
               + self._tf_input_pipeline.threads


class NoiseAugmentComponent(tensorpack.dataflow.ProxyDataFlow):

    def __init__(self, df, index, cfg, verbose=False):
        super(NoiseAugmentComponent, self).__init__(df)
        self._index = index
        if verbose:
            logger.info("Determining data augmentation")
        operation = cfg.get("operation", "add")
        if operation == "add":
            self._augment_op = np.add
        elif operation == "multiply":
            self._augment_op = np.multiply
        else:
            raise RuntimeError("Unknown augmentation operation for noise: {}".format(operation))
        noise_type = cfg.get("type", "normal")
        if noise_type == "normal":
            mean = cfg.get("mean", 0)
            stddev = cfg.get("stddev", 1)
            self._noise_fn = lambda x: np.random.normal(mean, stddev, size=x.shape).astype(x.dtype)
        elif noise_type == "uniform":
            low = cfg.get("low", 0)
            high = cfg.get("high", 1)
            self._noise_fn = lambda x: np.random.uniform(low, high, size=x.shape).astype(x.dtype)
        else:
            raise RuntimeError("Unknown augmentation noise type: {}".format(noise_type))

    def _augment_component(self, comp):
        noise = self._noise_fn(comp)
        augmented_comp = self._augment_op(comp, noise)
        return augmented_comp

    def get_data(self):
        for dp in self.ds.get_data():
            # Copy datapoint list
            augmented_dp = dp.copy()
            augmented_dp[self._index] = self._augment_component(augmented_dp[self._index])
            yield augmented_dp


class InputAndTargetDataFlow(tensorpack.dataflow.DataFlow):

    def __init__(self, data_path, data_cfg, shuffle_lmdb=False,
                 random_start_position=False, verbose=False, repeats=-1,
                 override_data_stats=None):
        logger.info("Setting up InputAndTargetDataflow")
        self._data_stats = override_data_stats
        if data_cfg.type == "lmdb":
            use_prefetching = data_cfg.get("use_prefetching", False)
            prefetch_queue_size = data_cfg.get("prefetch_queue_size", 10)
            prefetch_count = data_cfg.get("prefetch_process_count", 4)
            self._lmdb_df = tensorpack_utils.AutoLMDBData(
                data_path, auto_unbatch=False, shuffle=shuffle_lmdb, random_start_position=random_start_position,
                use_prefetching=use_prefetching, finish_prefetching=False,
                prefetch_queue_size=prefetch_queue_size, prefetch_count=prefetch_count)
            # Get data stats
            if self._data_stats is None:
                self._data_stats = self._lmdb_df.get_metadata("stats")
            assert self._data_stats is not None

            df = self._lmdb_df
            self._input_and_target_retriever = InputAndTargetFromData(data_cfg, self._data_stats,
                                                                      batch_data=False, verbose=verbose)
            df = tensorpack.dataflow.MapData(
                df,
                lambda data_dict: [self._input_and_target_retriever.get_input_from_data(data_dict[0]),
                                   self._input_and_target_retriever.get_target_from_data(data_dict[0])])
            for index, data_name in enumerate(["input", "target"]):
                if "augmentation" in data_cfg[data_name]:
                    if "noise" in data_cfg[data_name]["augmentation"]:
                        df = NoiseAugmentComponent(df, index, data_cfg[data_name]["augmentation"]["noise"], verbose=verbose)
            df = tensorpack.dataflow.MapData(
                df,
                lambda samples: [self._input_and_target_retriever.normalize_input(samples[0]),
                                 self._input_and_target_retriever.normalize_target(samples[1])])
            if use_prefetching:
                self._sample_prefetch_df = tensorpack_utils.PrefetchDataZMQ(df, prefetch_count, hwm=prefetch_queue_size)
                df = self._sample_prefetch_df
            else:
                self._sample_prefetch_df = None
            self._sample_prefetch_df = None
            max_num_batches = data_cfg.get("max_num_batches", -1)
            if max_num_batches > 0:
                df = tensorpack_utils.FixedSizeData(df, max_num_batches)
            self._batch_df_size = df.size()
            df = tensorpack.dataflow.RepeatedData(df, repeats)
            self._batch_df = df
            # TODO: This is ugly and brittle.
            self._batch_df_reset_state = self._batch_df.reset_state
            self._batch_df.reset_state = self._reset_batch_dataflow_state
            self._unbatch_df = tensorpack_utils.UnbatchData(df, total_size=self._batch_df_size)
        else:
            raise RuntimeError("Unknown data type: {}".format(data_cfg.type))
        if data_cfg.get("fake_constant_data", False) or data_cfg.get("fake_random_data", False):
            input_shape = data_cfg.fake_input_shape
            target_shape = data_cfg.fake_target_shape
            input_dtype = np.float32
            target_dtype = np.float32
            self._batch_df = tensorpack.dataflow.FakeData([[512] + input_shape, [512] + target_shape],
                                                          size=10000, random=data_cfg.get("fake_random_data", False),
                                                          dtype=[input_dtype, target_dtype])
            self._unbatch_df = tensorpack.dataflow.FakeData([input_shape, target_shape],
                                                            size=10000, random=data_cfg.get("fake_random_data", False),
                                                            dtype=[input_dtype, target_dtype])

    def _reset_batch_dataflow_state(self):
        self.start()
        self._batch_df_reset_state()

    def get_batch_dataflow(self):
        return self._batch_df

    def start(self):
        # Make sure we start the LMDB db prefetcher before the other data fetchers. This way all forks
        # start from the main process (this current one) and we can clean them up properly
        logger.info("Starting lmdb process")
        self._lmdb_df.start()
        if self._sample_prefetch_df is not None:
            logger.info("Starting prefetch processes")
            self._sample_prefetch_df.start()
        logger.info("Done")

    def reset_state(self):
        self.start()
        self._unbatch_df.reset_state()

    def get_data(self):
        for data_point in self._unbatch_df.get_data():
            yield data_point

    def size(self):
        return self._batch_df_size * self._lmdb_df.batch_size()

    def get_data_stats(self):
        return self._data_stats

    def get_sample_stats(self):
        return [self._input_and_target_retriever.input_stats, self._input_and_target_retriever.target_stats]

    @property
    def input_and_target_retriever(self):
        return self._input_and_target_retriever


class TFDataFlowPipeline(object):

    def __init__(self, dataflow, shapes, dtypes, sess, coord,
                 batch_size, tf_cfg, is_training, sample_stats=None, is_batch_dataflow=False):
        assert len(shapes) == 2 or len(shapes) == 3, "Only expecting input, target and weight tensor dtypes"
        assert len(dtypes) == 2 or len(dtypes) == 3, "Only expecting input, target and weight tensor shapes"
        sample_queue_capacity = tf_cfg.get("sample_queue_capacity", 2048)
        sample_queue_min_after_dequeue = tf_cfg.get("sample_queue_min_after_dequeue", 1024)
        logger.info("Creating TF queue with capacity {}".format(sample_queue_capacity))
        if is_training:
            self._tf_queue = tf.RandomShuffleQueue(
                capacity=sample_queue_capacity,
                min_after_dequeue=sample_queue_min_after_dequeue,
                dtypes=dtypes,
                shapes=shapes)
        else:
            self._tf_queue = tf.FIFOQueue(
                capacity=sample_queue_capacity // 2,
                dtypes=dtypes,
                shapes=shapes)

        self._dataflow = dataflow
        if is_batch_dataflow:
            self._df_to_tf_bridge = tensorpack_utils.BatchDataFlowToTensorflowBridge(
                self._dataflow, self._tf_queue, sess, coord)
        else:
            self._df_to_tf_bridge = tensorpack_utils.DataFlowToTensorflowBridge(
                self._dataflow, self._tf_queue, sess, coord)

        if sample_stats[0]["mean"] is not None:
            self._input_mean_tf = tf.constant(sample_stats[0]["mean"])
            self._input_stddev_tf = tf.constant(sample_stats[0]["stddev"])
        else:
            self._input_mean_tf = None
            self._input_stddev_tf = None
        if sample_stats[1]["mean"] is not None:
            self._target_mean_tf = tf.constant(sample_stats[1]["mean"])
            self._target_stddev_tf = tf.constant(sample_stats[1]["stddev"])
        else:
            self._target_mean_tf = None
            self._target_stddev_tf = None

        self._tensors = self._tf_queue.dequeue()
        self._tensors_batch = self._tf_queue.dequeue_many(batch_size)

        # For performance tuning
        # self._tensors = [tf.ones(shape=shape, dtype=dtype) for dtype, shape in zip(dtypes, shapes)]
        # self._tensors_batch = [tf.ones(shape=(batch_size,) + shape, dtype=dtype) for dtype, shape in zip(dtypes, shapes)]

        self._input_tf = self._tensors[0]
        self._target_tf = self._tensors[1]
        if len(self._tensors) > 2:
            self._weight_tf = self._tensors[2]
        else:
            self._weight_tf = None

        self._input_batch_tf = self._tensors_batch[0]
        self._target_batch_tf = self._tensors_batch[1]
        if len(self._tensors) > 2:
            self._weight_batch_tf = self._tensors_batch[2]
        else:
            self._weight_batch_tf = None

    def start(self):
        self._dataflow.reset_state()
        self._df_to_tf_bridge.start()

    def stop(self):
        self._df_to_tf_bridge.stop()

    @property
    def tf_queue(self):
        return self._tf_queue

    @property
    def tensors(self):
        return self._tensors

    @property
    def tensors_batch(self):
        return self._tensors_batch

    @property
    def input_tf(self):
        return self._input_tf

    @property
    def input_batch_tf(self):
        return self._input_batch_tf

    @property
    def target_tf(self):
        return self._target_tf

    @property
    def target_batch_tf(self):
        return self._target_batch_tf

    @property
    def weight_tf(self):
        return self._weight_tf

    @property
    def weight_batch_tf(self):
        return self._weight_batch_tf

    @property
    def unnormalized_input_tf(self):
        if self._input_mean_tf is None:
            return self._input_tf
        else:
            return tf.add(tf.multiply(self._input_tf, self._input_stddev_tf), self._input_mean_tf)

    @property
    def unnormalized_target_tf(self):
        if self._target_mean_tf is None:
            return self._target_tf
        else:
            return tf.add(tf.multiply(self._target_tf, self._target_stddev_tf), self._target_mean_tf)

    @property
    def unnormalized_input_batch_tf(self):
        if self._input_mean_tf is None:
            return self._input_batch_tf
        else:
            return tf.add(tf.multiply(self._input_batch_tf, self._input_stddev_tf), self._input_mean_tf)

    @property
    def unnormalized_target_batch_tf(self):
        if self._target_mean_tf is None:
            return self._target_batch_tf
        else:
            return tf.add(tf.multiply(self._target_batch_tf, self._target_stddev_tf), self._target_mean_tf)

    @property
    def threads(self):
        return [self._df_to_tf_bridge.thread]
