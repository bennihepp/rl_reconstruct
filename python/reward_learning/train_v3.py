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
import input_pipeline
import configuration
from utils import argparse_bool
from attribute_dict import AttributeDict
from tensorflow.python.client import timeline
from RLrecon.utils import Timer


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
    # Write train test split to model storage path
    with file(os.path.join(args.store_path, "train_filenames"), "w") as fout:
        for filename in train_filenames:
            fout.write("{}\n".format(os.path.basename(filename)))
    with file(os.path.join(args.store_path, "test_filenames"), "w") as fout:
        for filename in test_filenames:
            fout.write("{}\n".format(os.path.basename(filename)))

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

    if not cfg.data.input_stats_filename:
        cfg.data.input_stats_filename = os.path.join(args.data_path, file_helpers.DEFAULT_HDF5_STATS_FILENAME)
    input_shape, get_input_from_record, target_shape, get_target_from_record = \
        input_pipeline.get_input_and_target_from_record_functions(cfg.data, all_filenames, verbose=True)

    if args.verbose:
        input_pipeline.print_data_stats(all_filenames[0], get_input_from_record, get_target_from_record)

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

    def parse_record(record):
        single_input = get_input_from_record(record)
        single_target = get_target_from_record(record)
        return single_input, single_target

    with tf.device("/cpu:0"):
        train_input_pipeline = input_pipeline.InputPipeline(
            sess, coord, train_filenames, parse_record,
            input_shape, target_shape, batch_size_per_device,
            cfg.tensorflow.cpu_train_queue_capacity,
            cfg.tensorflow.cpu_train_queue_min_after_dequeue,
            shuffle=True,
            num_threads=cfg.tensorflow.cpu_train_queue_threads,
            timeout=cfg.io.timeout,
            fake_constant_data=cfg.data.fake_constant_data,
            fake_random_data=cfg.data.fake_random_data,
            name="train",
            verbose=args.verbose)
        test_input_pipeline = input_pipeline.InputPipeline(
            sess, coord, test_filenames, parse_record,
            input_shape, target_shape, batch_size_per_device,
            cfg.tensorflow.cpu_test_queue_capacity,
            cfg.tensorflow.cpu_test_queue_min_after_dequeue,
            shuffle=False,
            num_threads=cfg.tensorflow.cpu_test_queue_threads,
            timeout=cfg.io.timeout,
            fake_constant_data=cfg.data.fake_constant_data,
            fake_random_data=cfg.data.fake_random_data,
            name="test",
            verbose=args.verbose)
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
                           keep_checkpoint_every_n_hours=cfg.io.keep_checkpoint_every_n_hours,
                           save_relative_paths=True)

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
                    and epoch % cfg.io.validation_interval == 0:
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
        coord.join(custom_threads, stop_grace_period_secs=cfg.io.timeout)


if __name__ == '__main__':
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--test-data-path', required=False, help='Test data path. If not provided data will be split.')
    parser.add_argument('--store-path', required=True, help='Store path.')
    parser.add_argument('--log-path', required=False, help='Log path.')
    parser.add_argument('--restore', type=argparse_bool, default=False, help='Whether to restore existing model.')
    parser.add_argument('--config', type=str, help='YAML configuration file.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')

    # Report and checkpoint saving
    parser.add_argument('--io.timeout', type=int, default=10 * 60)
    parser.add_argument('--io.validation_interval', type=int, default=10)
    parser.add_argument('--io.train_summary_interval', type=int, default=10)
    parser.add_argument('--io.model_summary_interval', type=int, default=10)
    parser.add_argument('--io.model_summary_num_batches', type=int, default=50)
    parser.add_argument('--io.checkpoint_interval', type=int, default=50)
    parser.add_argument('--io.keep_checkpoint_every_n_hours', type=float, default=1)
    parser.add_argument('--io.keep_n_last_checkpoints', type=int, default=5)
    parser.add_argument('--io.max_checkpoint_save_trials', type=int, default=5)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.gpu_ids', type=str,
                        help='GPU to be used (-1 for CPU). Comma separated list for multiple GPUs')
    parser.add_argument('--tf.strict_devices', type=argparse_bool, default=False,
                        help='Should the user be prompted to use fewer devices instead of failing.')
    parser.add_argument('--tf.multi_gpu_ps_id', type=int, help='Device to use as parameter server in Multi GPU mode')
    parser.add_argument('--tf.gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=1)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.cpu_train_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--tf.cpu_test_queue_capacity', type=int, default=1024 * 8)
    parser.add_argument('--tf.cpu_train_queue_min_after_dequeue', type=int, default=2000)
    parser.add_argument('--tf.cpu_test_queue_min_after_dequeue', type=int, default=0)
    parser.add_argument('--tf.cpu_train_queue_threads', type=int, default=4)
    parser.add_argument('--tf.cpu_test_queue_threads', type=int, default=1)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")
    parser.add_argument('--tf.create_tf_timeline', type=argparse_bool, default=False,
                        help="Generate tensorflow trace.")

    # Learning parameters
    parser.add_argument('--training.num_epochs', type=int, default=10000)
    parser.add_argument('--training.batch_size', type=int, default=64)
    parser.add_argument('--training.optimizer', type=str, default="adam")
    parser.add_argument('--training.max_grad_global_norm', type=float, default=1e3)
    parser.add_argument('--training.initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--training.learning_rate_decay_epochs', type=int, default=10)
    parser.add_argument('--training.learning_rate_decay_rate', type=float, default=0.96)
    parser.add_argument('--training.learning_rate_decay_staircase', type=argparse_bool, default=False)

    # Train test split
    parser.add_argument('--data.train_test_split_ratio', type=float, default=4.0)
    parser.add_argument('--data.train_test_split_shuffle', type=argparse_bool, default=False)
    parser.add_argument('--data.max_num_train_files', type=int, default=-1)
    parser.add_argument('--data.max_num_test_files', type=int, default=-1)
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
    parser.add_argument('--data.fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--data.fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

    args = parser.parse_args()

    run(args)
