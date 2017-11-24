#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import traceback
import models
import numpy as np
import yaml
import tensorflow as tf
import input_pipeline
import configuration
from pybh.utils import argparse_bool
from pybh.attribute_dict import AttributeDict
from pybh import log_utils
from pybh import math_utils
from pybh import hdf5_utils
from pybh import tf_utils


logger = log_utils.get_logger("reward_learning/train")


def run(args):
    # Read config file
    topic_cmdline_mappings = {"tensorflow": "tf"}
    topics = ["tensorflow", "io", "training", "data"]
    cfg = configuration.get_config_from_cmdline(
        args, topics, topic_cmdline_mappings)
    if args.config is not None:
        with open(args.config, "r") as config_file:
            tmp_cfg = yaml.load(config_file)
            configuration.update_config_from_other(cfg, tmp_cfg)

    # Read model config
    if "model" in cfg:
        model_config = cfg["model"]
    elif args.model_config is not None:
        model_config = {}
    else:
        logger.fatal("ERROR: Model configuration must be in general config file or provided in extra config file.")
        import sys
        sys.exit(1)

    if args.model_config is not None:
        with open(args.model_config, "r") as config_file:
            tmp_model_config = yaml.load(config_file)
            configuration.update_config_from_other(model_config, tmp_model_config)

    cfg = AttributeDict.convert_deep(cfg)
    model_config = AttributeDict.convert_deep(model_config)

    if args.hdf5_data_stats_path is not None:
        logger.info("Loading data stats from HDF5 file")
        data_stats_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(args.hdf5_data_stats_path)
    else:
        data_stats_dict = None
    if args.use_train_data or data_stats_dict is None:
        logger.info("Creating train dataflow")
        train_dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.train_path, cfg.data, shuffle_lmdb=args.shuffle,
                                                               override_data_stats=data_stats_dict, verbose=True)
        data_stats_dict = train_dataflow.get_data_stats()
        if args.use_train_data:
            dataflow = train_dataflow
    if not args.use_train_data:
        assert cfg.data.test_path is not None, "Test data path has to be specified if not using train data"
        logger.info("Creating test dataflow")
        dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.test_path, cfg.data, shuffle_lmdb=args.shuffle,
                                                         override_data_stats=data_stats_dict,
                                                         verbose=True)

    logger.info("# samples in dataset: {}".format(dataflow.size()))

    logger.info("Input and target shapes:")
    dataflow.reset_state()
    first_sample = next(dataflow.get_data())
    tensor_shapes = [tensor.shape for tensor in first_sample]
    tensor_dtypes = [tensor.dtype for tensor in first_sample]
    logger.info("  Shape of input: {}".format(first_sample[0].shape))
    logger.info("  Type of input: {}".format(first_sample[0].dtype))
    logger.info("  Shape of target: {}".format(first_sample[1].shape))
    logger.info("  Type of target: {}".format(first_sample[1].dtype))

    # Create tensorflow session
    logger.info("Creating tensorflow session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.tensorflow.gpu_memory_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
    tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
    tf_config.log_device_placement = cfg.tensorflow.log_device_placement
    with tf.Session(config=tf_config) as sess:
        coord = tf.train.Coordinator()

        # def signal_handler(signal, frame):
        #     sess.close()
        #
        # batch_size = 1
        #
        # sample_stats = dataflow.get_sample_stats()
        # pipeline = input_pipeline.TFDataFlowPipeline(
        #     dataflow.get_batch_dataflow(), tensor_shapes, tensor_dtypes, sess, coord, batch_size,
        #         cfg.tensorflow, is_training=False, sample_stats=sample_stats, is_batch_dataflow=True)
        #
        # Create model
        #
        # with tf.device("/gpu:0"):
        #     with tf.variable_scope("model"):
        #             model = models.Model(model_config,
        #                                  pipeline.tensors_batch[0],
        #                                  pipeline.tensors[1],
        #                                  is_training=False,
        #                                  verbose=args.verbose)

        input_placeholder = tf.placeholder(dtype=tensor_dtypes[0], shape=(1,) + tensor_shapes[0], name="Input")
        target_placeholder = tf.placeholder(dtype=tensor_dtypes[1], shape=(1,) + tensor_shapes[1], name="Target")
        gpu_device_name = tf_utils.gpu_device_name()
        print(tf_utils.get_available_cpu_ids(), tf_utils.get_available_cpu_names())
        print(tf_utils.get_available_gpu_ids(), tf_utils.get_available_gpu_names())
        print(gpu_device_name)
        with tf.device(gpu_device_name):
            with tf.variable_scope("model"):
                model = models.Model(model_config,
                                     input_placeholder,
                                     target_placeholder,
                                     is_training=False,
                                     verbose=args.verbose)

    try:
        saver = tf.train.Saver(model.global_variables)

        if args.check_numerics:
            # Run numeric checks on all model checkpoints
            if args.checkpoint is None:
                ckpt = tf.train.get_checkpoint_state(args.model_dir)
                checkpoint_paths = ckpt.all_model_checkpoint_paths
            else:
                checkpoint_path = os.path.join(args.model_dir, args.checkpoint)
                checkpoint_paths = [checkpoint_path]
            for checkpoint_path in checkpoint_paths:
                if args.verbose:
                    logger.info("Checking numerics on model checkpoint {}".format(checkpoint_path))
                saver.restore(sess, checkpoint_path)
                for var in model.variables:
                    if args.verbose:
                        logger.info("  Checking tensor {}".format(var.name))
                    sess.run(tf.check_numerics(var, "Numeric check for tensor {} failed".format(var.name)))
            return

        # Restore model
        if args.checkpoint is None:
            logger.info("Reading latest checkpoint from {}".format(args.model_dir))
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            if ckpt is None:
                raise IOError("No previous checkpoint found at {}".format(args.model_dir))
            else:
                logger.info('Found previous checkpoint... restoring')
                checkpoint_path = ckpt.model_checkpoint_path
            saver.restore(sess, checkpoint_path)
        else:
            checkpoint_path = os.path.join(args.model_dir, args.checkpoint)
        if checkpoint_path is not None:
            logger.info("Trying to restore model from checkpoint {}".format(checkpoint_path))
            saver.restore(sess, checkpoint_path)

        # custom_threads = []
        # pipeline.start()
        # custom_threads.extend(pipeline.threads)
        # # Start data provider threads
        # custom_threads.extend(tf.train.start_queue_runners(sess=sess))

        sess.graph.finalize()

        # Running statistics
        stats = None

        denorm_target_list = []
        denorm_output_list = []

        logger.info("Starting evaluating")
        for i, (input, target) in enumerate(dataflow.get_data()):
            if args.verbose:
                logger.info("  sample # {}".format(i))

            if stats is None:
                stats = AttributeDict()
                stats.output = math_utils.SinglePassStatistics(target.shape)
                stats.target = math_utils.SinglePassStatistics(target.shape)
                stats.diff = math_utils.SinglePassStatistics(target.shape)
                stats.squared_diff = math_utils.SinglePassStatistics(target.shape)
                stats.loss = math_utils.SinglePassStatistics(target.shape)

            denorm_target = dataflow.input_and_target_retriever.denormalize_target(target)

            input_batch = input[np.newaxis, ...]
            target_batch = target[np.newaxis, ...]
            loss_v, loss_min_v, loss_max_v, output_batch = sess.run(
                [model.loss, model.loss_min, model.loss_max, model.output],
                feed_dict={
                   input_placeholder: input_batch,
                   target_placeholder: target_batch
                }
            )
            output = output_batch[0, ...]
            denorm_output = dataflow.input_and_target_retriever.denormalize_target(output)
            diff = denorm_output - denorm_target
            squared_diff = np.square(diff)
            if args.verbose:
                logger.info("Output={}, Target={}, Diff={}, Diff^2={}".format(denorm_output, denorm_target, diff, squared_diff))
                logger.info("  loss: {}, min loss: {}, max loss: {}".format(loss_v, loss_min_v, loss_max_v))
            if diff > 80:
                import time
                time.sleep(5)
            # Update stats
            stats.output.add_value(denorm_output)
            stats.target.add_value(denorm_target)
            stats.diff.add_value(diff)
            stats.squared_diff.add_value(squared_diff)
            stats.loss.add_value(loss_v)

            denorm_output_list.append(denorm_output)
            denorm_target_list.append(denorm_target)

            if i % 100 == 0 and i > 0:
                logger.info("-----------")
                logger.info("Statistics after {} samples:".format(i + 1))
                for key in stats:
                    logger.info("  {:s}: mean={:.4f}, stddev={:.4f}, min={:.4f}, max={:.4f}".format(
                        key, stats[key].mean[0], stats[key].stddev[0], float(stats[key].min), float(stats[key].max)))
                    logger.info("-----------")

                import scipy.stats
                correlation, pvalue = scipy.stats.pearsonr(np.array(denorm_target_list), np.array(denorm_output_list))
                logger.info("Pearson correlation: {} [p={}]".format(correlation, pvalue))
                correlation, pvalue = scipy.stats.spearmanr(np.array(denorm_target_list), np.array(denorm_output_list))
                logger.info("Spearman correlation: {} [p={}]".format(correlation, pvalue))
                obj = {"a": np.array(denorm_target_list), "b": np.array(denorm_output_list)}
                np.savez("spearman.npz", **obj)
                hdf5_utils.write_numpy_dict_to_hdf5_file("spearman.hdf5", obj)

            if args.visualize:
                import visualization
                fig = 1
                fig = visualization.plot_grid(input[..., 2], input[..., 3], title_prefix="input", show=False, fig_offset=fig)
                # fig = visualization.plot_grid(record.in_grid_3d[..., 6], record.in_grid_3d[..., 7], title_prefix="in_grid_3d", show=False, fig_offset=fig)
                visualization.show(stop=True)

    except Exception as exc:
        logger.info("Exception in evaluation oop: {}".format(exc))
        traceback.print_exc()
        coord.request_stop(exc)
        raise exc
    finally:
        logger.info("Requesting stop")
        coord.request_stop()
        # pipeline.stop()
        # coord.join(custom_threads, stop_grace_period_secs=(2 * cfg.io.timeout))
        sess.close()


if __name__ == '__main__':
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--hdf5-data-stats-path', help='HDF5 data stats file.')
    parser.add_argument('--model-dir', required=True, help='Model path.')
    parser.add_argument('--checkpoint', help='Checkpoint to restore.')
    parser.add_argument('--config', type=str, help='YAML configuration.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')
    parser.add_argument('--visualize', type=argparse_bool, default=False)
    parser.add_argument('--shuffle', type=argparse_bool, default=True)
    parser.add_argument('--use-train-data', type=argparse_bool, default=True)
    parser.add_argument('--check-numerics', type=argparse_bool, default=False)

    # IO
    parser.add_argument('--io.timeout', type=int, default=10 * 60)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=1)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")

    # Data parameters
    parser.add_argument('--data.train_path', required=True, help='Data path.')
    parser.add_argument('--data.test_path', required=False, help='Data path.')
    parser.add_argument('--data.type', default="lmdb", help='Type of data storage.')

    args = parser.parse_args()

    run(args)
