#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import exceptions
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
import input_pipeline
import configuration
import visualization
from attribute_dict import AttributeDict
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


def run(args):
    # Read config file
    topic_cmdline_mappings = {"tensorflow": "tf"}
    topics = ["tensorflow", "io", "training", "data"]
    cfg = configuration.get_config_from_cmdline(
        args, topics, topic_cmdline_mappings)
    if args.config is not None:
        with file(args.config, "r") as config_file:
            tmp_cfg = yaml.load(config_file)
            configuration.update_config_from_other(cfg, tmp_cfg)

    # Read model config
    if "model" in cfg:
        model_config = cfg["model"]
    elif args.model_config is not None:
        model_config = {}
    else:
        print("ERROR: Model configuration must be in general config file or provided in extra config file.")
        import sys
        sys.exit(1)

    if args.model_config is not None:
        with file(args.model_config, "r") as config_file:
            tmp_model_config = yaml.load(config_file)
            configuration.update_config_from_other(model_config, tmp_model_config)

    cfg = AttributeDict.convert_deep(cfg)
    model_config = AttributeDict.convert_deep(model_config)

    filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
    filenames = sorted(list(filename_generator))
    np.random.shuffle(filenames)

    if len(filenames) == 0:
        raise RuntimeError("No dataset file")
    else:
        print("Found {} dataset files".format(len(filenames)))

    input_shape, get_input_from_record, target_shape, get_target_from_record = \
        input_pipeline.get_input_and_target_from_record_functions(cfg.data, filenames, args.verbose)

    if args.verbose:
        input_pipeline.print_data_stats(filenames[0], get_input_from_record, get_target_from_record)

    # Configure tensorflow
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = cfg.tensorflow.gpu_memory_fraction
    tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
    tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
    tf_config.log_device_placement = cfg.tensorflow.log_device_placement

    batch_size = 1

    sess = tf.Session(config=tf_config)

    def parse_record(record):
        single_input = get_input_from_record(record)
        single_target = get_target_from_record(record)
        return single_input, single_target

    # Create model
    if len(tf_utils.get_available_gpu_names()) > 0:
        device_name = tf_utils.get_available_gpu_names()[0]
    else:
        device_name = tf_utils.cpu_device_name()
    with tf.device(device_name):
        input_placeholder = tf.placeholder(tf.float32, shape=[batch_size] + list(input_shape), name="in_input")
        target_placeholder = tf.placeholder(tf.float32, shape=[batch_size] + list(target_shape), name="in_target")
        with tf.variable_scope("model"):
            model = models.Model(model_config,
                                 input_placeholder,
                                 target_placeholder,
                                 is_training=False,
                                 verbose=args.verbose)

    gradients_and_variables = list(zip(model.gradients, model.variables))
    if args.verbose:
        print("Tensorflow variables:")
        for var in tf.global_variables():
            print("  {}: {}".format(var.name, var.shape))
    print("Model variables:")
    for grad, var in gradients_and_variables:
        print("  {}: {}".format(var.name, var.shape))

    # Print model size etc.
    model_size = 0
    model_grad_size = 0
    for grad, var in gradients_and_variables:
        model_size += np.sum(tf_utils.variable_size(var))
        if grad is not None:
            model_grad_size += np.sum(tf_utils.variable_size(grad))
    print("Model variables: {} ({} MB)".format(model_size, model_size / 1024. / 1024.))
    print("Model gradients: {} ({} MB)".format(model_grad_size, model_grad_size / 1024. / 1024.))
    for name, module in model.modules.iteritems():
        module_size = np.sum([tf_utils.variable_size(var) for var in module.variables])
        print("Module {} variables: {} ({} MB)".format(name, module_size, module_size / 1024. / 1024.))

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(model.global_variables, save_relative_paths=True)

    # Restore model
    if args.checkpoint is None:
        print("Reading latest checkpoint from {}".format(args.model_path))
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        if ckpt is None:
            raise exceptions.IOError("No previous checkpoint found at {}".format(args.model_path))
        print('Found previous checkpoint... restoring')
        checkpoint_path = ckpt.model_checkpoint_path
    else:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint)
    print("Trying to restore model from checkpoint {}".format(checkpoint_path))
    saver.restore(sess, checkpoint_path)

    sess.graph.finalize()

    print("Starting evaluating")
    for filename in filenames:
        print("Evaluating file {}".format(filename))
        records = data_record.read_hdf5_records_v3_as_list(filename)
        np.random.shuffle(records)
        for i, record in enumerate(records):
            print("  record # {}".format(i))
            input, target = parse_record(record)
            input_batch = input[np.newaxis, ...]
            target_batch = target[np.newaxis, ...]
            loss_v, loss_min_v, loss_max_v, output_batch = sess.run(
                [model.loss, model.loss_min, model.loss_max, model.output],
                feed_dict={
                    input_placeholder: input_batch,
                    target_placeholder: target_batch
                })
            output = output_batch[0, ...]
            dummy = 0.5 * np.ones(output.shape)
            print(np.mean(np.square(dummy - target)))
            print(np.mean(np.square(output - target)))
            print("  loss: {}, min loss: {}, max loss: {}".format(loss_v, loss_min_v, loss_max_v))
            if args.visualize:
                fig = 1
                fig = visualization.plot_grid(input[..., 0], input[..., 1], title_prefix="input", show=False, fig_offset=fig)
                fig = visualization.plot_grid(target[..., 0], target[..., 1], title_prefix="target", show=False, fig_offset=fig)
                fig = visualization.plot_grid(output[..., 0], output[..., 1], title_prefix="output", show=False, fig_offset=fig)
                diff = output - target
                fig = visualization.plot_diff_grid(diff[..., 0], diff[..., 1], title_prefix="diff", show=False, fig_offset=fig)
                visualization.show(stop=True)


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
    parser.add_argument('--model-path', required=True, help='Model path.')
    parser.add_argument('--checkpoint', help='Checkpoint to restore.')
    parser.add_argument('--config', type=str, help='YAML configuration.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')
    parser.add_argument('--visualize', type=argparse_bool, default=False)

    # IO
    parser.add_argument('--io.timeout', type=int, default=10 * 60)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=1)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")

    # Data parameters
    parser.add_argument('--data.input_id', type=str, default="in_grid_3d")
    parser.add_argument('--data.target_id', type=str, default="prob_reward")
    parser.add_argument('--data.input_stats_filename', type=str)
    parser.add_argument('--data.obs_levels_to_use', type=str)
    parser.add_argument('--data.subvolume_slice_x', type=str)
    parser.add_argument('--data.subvolume_slice_y', type=str)
    parser.add_argument('--data.subvolume_slice_z', type=str)
    parser.add_argument('--data.normalize_input', type=argparse_bool, default=True)
    parser.add_argument('--data.normalize_target', type=argparse_bool, default=False)
    parser.add_argument('--fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

    args = parser.parse_args()

    run(args)
