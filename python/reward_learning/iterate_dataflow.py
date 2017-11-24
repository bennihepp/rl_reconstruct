#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Workaround for segmentation fault for some versions when ndimage is imported after tensorflow.
import scipy.ndimage as nd

import os
import sys
import argparse
import time
import models
import signal
import numpy as np
from pybh import pybh_yaml as yaml
import tensorflow as tf
import tensorflow.contrib.memory_stats as tf_memory_stats
from tensorflow.python.framework.errors_impl import InvalidArgumentError as TFInvalidArgumentError
from pybh import tf_utils, log_utils
import data_provider
import input_pipeline
import configuration
import traceback
from pybh.utils import argparse_bool
from pybh.attribute_dict import AttributeDict
from tensorflow.python.client import timeline
from pybh.utils import Timer, logged_time_measurement
from pybh import progressbar


logger = log_utils.get_logger("reward_learning/train")


def run(args):
    # Read config file
    topics = ["data"]
    cfg = configuration.get_config_from_cmdline(args, topics)
    if args.config is not None:
        with open(args.config, "r") as config_file:
            tmp_cfg = yaml.load(config_file)
            configuration.update_config_from_other(cfg, tmp_cfg)

    cfg = AttributeDict.convert_deep(cfg)

    logger.info("Creating train dataflow")
    print(cfg.data.max_num_samples)
    train_dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.train_path, cfg.data,
                                                           shuffle_lmdb=args.shuffle,
                                                           repeats=1,
                                                           verbose=True)
    input_and_target_retriever = train_dataflow.input_and_target_retriever

    # Important: Stats for normalizing should be the same for both train and test dataset
    input_stats, target_stats = train_dataflow.get_sample_stats()
    # logger.info("Input statistics: {}".format(input_stats))
    # for x in range(input_stats["mean"].shape[0]):
    #     for y in range(input_stats["mean"].shape[1]):
    #         logger.info("mean input x={}, y={}: {} [{}]".format(x, y, input_stats["mean"][x, y, :],
    #                                                             input_stats["stddev"][x, y, :]))
    logger.info("Target statistics: {}".format(target_stats))

    if args.use_train_data:
        logger.info("Using train dataflow")
        dataflow = train_dataflow
    else:
        assert cfg.data.test_path is not None, "Test data path has to be specified if not using train data"
        logger.info("Creating test dataflow")
        dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.test_path, cfg.data, shuffle_lmdb=args.shuffle,
                                                         repeats=1,
                                                         override_data_stats=train_dataflow.get_data_stats(),
                                                         verbose=True)
    if args.shuffle:
        import tensorpack
        dataflow = tensorpack.dataflow.LocallyShuffleData(dataflow, 1024)

    logger.info("# samples in dataset: {}".format(dataflow.size()))

    logger.info("Input and target shapes:")
    dataflow.reset_state()
    first_sample = next(dataflow.get_data())
    tensor_shapes = [tensor.shape for tensor in first_sample] + [(1,)]
    tensor_dtypes = [tensor.dtype for tensor in first_sample] + [np.float32]
    logger.info("  Shape of input: {}".format(first_sample[0].shape))
    logger.info("  Type of input: {}".format(first_sample[0].dtype))
    logger.info("  Shape of target: {}".format(first_sample[1].shape))
    logger.info("  Type of target: {}".format(first_sample[1].dtype))
    logger.info("  Shape of weights: {}".format(tensor_shapes[2]))
    logger.info("  Type of weights: {}".format(tensor_dtypes[2]))

    logger.info("Iterating dataflow")

    if cfg.data.max_num_samples is not None and cfg.data.max_num_samples > 0:
        from pybh import tensorpack_utils
        dataflow = tensorpack_utils.FixedSizeData(dataflow, cfg.data.max_num_samples)

    dataflow.reset_state()
    for i, (input, target) in enumerate(dataflow.get_data()):
        # if args.verbose:
        logger.info("  sample # {}".format(i))

        denorm_input = input_and_target_retriever.denormalize_input(input)
        denorm_input = np.minimum(denorm_input, 1)
        denorm_input = np.maximum(denorm_input, 0)

        # if args.verbose:
        logger.info("Target={}, input mean={}, input stddev={}".format(target, np.mean(input), np.std(input)))
        logger.info("Restored target={}".format(input_and_target_retriever.denormalize_target(target)))

        # if i % 100 == 0:
        #     logger.info("-----------")
        #     logger.info("Statistics after {} samples:".format(i + 1))

        if args.visualize:
            # import visualization
            #
            # fig = 1
            # fig = visualization.plot_grid(input[..., 2], input[..., 3], title_prefix="input", show=False, fig_offset=fig)
            # # fig = visualization.plot_grid(record.in_grid_3d[..., 6], record.in_grid_3d[..., 7], title_prefix="in_grid_3d", show=False, fig_offset=fig)
            # visualization.show(stop=True)

            import cv2
            from matplotlib import pyplot as plt
            img_size = args.visualization_size
            images = []
            logger.info("input.shape: {}".format(input.shape))
            logger.info(np.min(denorm_input))
            logger.info(np.max(denorm_input))
            for channel in range(input.shape[-1]):
                img = denorm_input[:, :, input.shape[2] // 2, channel]
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                images.append(img)

            line_size = 16
            horizontal_line = 1 * np.ones((line_size, img_size))
            num_levels = input.shape[-1] // 2
            img_pyramid_left = np.zeros((img_size, img_size))
            img_pyramid_left[...] = images[0]
            for level in range(1, num_levels):
                channel = level * 2
                img_pyramid_left = np.vstack([img_pyramid_left, horizontal_line])
                img_pyramid_left = np.vstack([img_pyramid_left, images[channel]])

            img_pyramid_right = np.zeros((img_size, img_size))
            img_pyramid_right[...] = images[1]
            for level in range(1, num_levels):
                channel = level * 2 + 1
                img_pyramid_right = np.vstack([img_pyramid_right, horizontal_line])
                img_pyramid_right = np.vstack([img_pyramid_right, images[channel]])

            vertical_line = 1 * np.ones((img_pyramid_left.shape[0], line_size))
            img_pyramid = np.hstack([img_pyramid_left, vertical_line, img_pyramid_right])
            logger.info("Pyramid shape: {}".format(img_pyramid.shape))
            cv2.imshow("pyramid", img_pyramid.T)
            if args.save_visualization:
                img_filename = "image_pyramid_{}.png".format(i)
                cv2.imwrite(img_filename, 255 * img_pyramid.T)
            cv2.waitKey()


if __name__ == '__main__':
    np.set_printoptions(threshold=50)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Set verbosity level.')
    parser.add_argument('--verbose-model', action='count',
                        default=0, help='Set verbosity level for model generation.')
    parser.add_argument('--config', type=str, help='YAML configuration file.')
    parser.add_argument('--visualize', type=argparse_bool, default=True)
    parser.add_argument('--visualization-size', type=int, default=256)
    parser.add_argument('--save-visualization', type=argparse_bool, default=False)
    parser.add_argument('--shuffle', type=argparse_bool, default=True)
    parser.add_argument('--use-train-data', type=argparse_bool, default=True)

    # Data parameters
    parser.add_argument('--data.train_path', required=True, help='Train data path.')
    parser.add_argument('--data.test_path', required=False, help='Test data path.')
    parser.add_argument('--data.type', default="lmdb", help='Type of data storage.')
    parser.add_argument('--data.max_num_batches', type=int, default=-1)
    parser.add_argument('--data.max_num_samples', type=int, default=-1)
    parser.add_argument('--data.use_prefetching', type=argparse_bool, default=False)
    parser.add_argument('--data.prefetch_process_count', type=int, default=2)
    parser.add_argument('--data.prefetch_queue_size', type=int, default=10)

    args = parser.parse_args()

    run(args)
