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
import scipy.stats
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


def make_image_summaries(cfg, batch, name=None):
    if name is None:
        name = batch.name
    if "slices" in cfg:
        image_summary_slices = []
        # Batch dimension
        image_summary_slices.append(slice(None))
        for slice_cfg in cfg.slices:
            if len(slice_cfg) == 0:
                image_summary_slices.append(slice(None))
            else:
                image_summary_slices.append(slice(slice_cfg[0], slice_cfg[1]))
    else:
        image_summary_slices = None
    if "channels" in cfg:
        image_summary_channels = cfg.channels
    else:
        image_summary_channels = range(batch.get_shape().as_list()[-1])
    if "clip_min" in cfg:
        if isinstance(cfg.clip_min, list):
            image_clip_min = cfg.clip_min
        else:
            image_clip_min = [cfg.clip_min] * len(image_summary_channels)
    else:
        image_clip_min = [0.0] * len(image_summary_channels)
    if "clip_max" in cfg:
        if isinstance(cfg.clip_max, list):
            image_clip_max = cfg.clip_max
        else:
            image_clip_max = [cfg.clip_max] * len(image_summary_channels)
    else:
        image_clip_max = [np.finfo(np.float32).max] * len(image_summary_channels)

    image_summaries = []
    for i, channel in enumerate(image_summary_channels):
        clip_min = image_clip_min[i]
        clip_max = image_clip_max[i]
        if image_summary_slices is None:
            image_batch = batch[..., channel]
        else:
            image_batch = batch[image_summary_slices + [channel]]
        # Reintroduce last channel dimension
        image_batch = image_batch[..., None]

        if "reduce" in cfg:
            image_summary_reduce = cfg.reduce
            # Go through axis in reverse order and reduce if necessary.
            # Reverse order is important because a reduced axis will be removed from the tensor.
            for i, reduction_str in reversed(list(enumerate(image_summary_reduce))):
                axis = i + 1
                if reduction_str == "none" or reduction_str == "":
                    reduce_fn = None
                elif reduction_str == "sum":
                    reduce_fn = tf.reduce_sum
                elif reduction_str == "prod":
                    reduce_fn = tf.reduce_prod
                elif reduction_str == "mean":
                    reduce_fn = tf.reduce_mean
                elif reduction_str == "min":
                    reduce_fn = tf.reduce_min
                elif reduction_str == "max":
                    reduce_fn = tf.reduce_max
                else:
                    raise RuntimeError("Unknown reduction type: {}".format(reduction_str))
                if reduce_fn is not None:
                    image_batch = reduce_fn(image_batch, axis=axis)

        logger.info("Image summary for channel {} will be clipped to [{}, {}]".format(channel, clip_min, clip_max))
        image_batch = tf.clip_by_value(image_batch, clip_min, clip_max)
        if cfg.get("clip_scale", False):
            image_batch = 255 * (image_batch - clip_min) / (clip_max - clip_min)
            image_batch = tf.cast(image_batch, tf.uint8)

        if "tile_multiples" in cfg:
            # Add batch and channel dimension to tile multiples
            tile_multiples = [1] + cfg.tile_multiples + [1]
            image_batch = tf.tile(image_batch, tile_multiples)

        if "resize" in cfg:
            new_size = cfg.resize
            resize_method_name = cfg.get("resize_method", "nearest")
            if resize_method_name == "nearest":
                resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif resize_method_name == "bilinear":
                resize_method = tf.image.ResizeMethod.BILINEAR
            elif resize_method_name == "bicubic":
                resize_method = tf.image.ResizeMethod.BICUBIC
            elif resize_method_name == "area":
                resize_method = tf.image.ResizeMethod.AREA
            else:
                raise RuntimeError("Unknown image resize method: {}".format(resize_method_name))
            image_batch = tf.image.resize_images(image_batch, new_size, method=resize_method)

        image_summary_count = cfg.get("image_summary_count", 16)
        image_summaries.append(tf.summary.image("image/{}/{}".format(name, channel), image_batch, image_summary_count))
    return image_summaries


def make_input_target_image_summaries(cfg, target_batch, predicted_batch):
    if "slices" in cfg:
        image_summary_slices = []
        # Batch dimension
        image_summary_slices.append(slice(None))
        for slice_cfg in cfg.slices:
            if len(slice_cfg) == 0:
                image_summary_slices.append(slice(None))
            else:
                image_summary_slices.append(slice(slice_cfg[0], slice_cfg[1]))
    else:
        image_summary_slices = None
    if "channels" in cfg:
        image_summary_channels = cfg.channels
    else:
        image_summary_channels = range(target_batch.get_shape().as_list()[-1])
    if "clip_min" in cfg:
        if isinstance(cfg.clip_min, list):
            image_clip_min = cfg.clip_min
        else:
            image_clip_min = [cfg.clip_min] * len(image_summary_channels)
    else:
        image_clip_min = [0.0] * len(image_summary_channels)
    if "clip_max" in cfg:
        if isinstance(cfg.clip_max, list):
            image_clip_max = cfg.clip_max
        else:
            image_clip_max = [cfg.clip_max] * len(image_summary_channels)
    else:
        image_clip_max = [np.finfo(np.float32).max] * len(image_summary_channels)

    image_summaries = []
    for i, channel in enumerate(image_summary_channels):
        clip_min = image_clip_min[i]
        clip_max = image_clip_max[i]
        if image_summary_slices is None:
            target_image_batch = target_batch[..., channel]
            predicted_image_batch = predicted_batch[..., channel]
        else:
            target_image_batch = target_batch[image_summary_slices + [channel]]
            predicted_image_batch = predicted_batch[image_summary_slices + [channel]]
        # Reintroduce last channel dimension
        target_image_batch = target_image_batch[..., None]
        predicted_image_batch = predicted_image_batch[..., None]
        abs_diff_image = tf.abs(target_image_batch - predicted_image_batch)

        if "reduce" in cfg:
            image_summary_reduce = cfg.reduce
            # Go through axis in reverse order and reduce if necessary.
            # Reverse order is important because a reduced axis will be removed from the tensor.
            for i, reduction_str in reversed(list(enumerate(image_summary_reduce))):
                axis = i + 1
                if reduction_str == "none" or reduction_str == "":
                    reduce_fn = None
                elif reduction_str == "sum":
                    reduce_fn = tf.reduce_sum
                elif reduction_str == "prod":
                    reduce_fn = tf.reduce_prod
                elif reduction_str == "mean":
                    reduce_fn = tf.reduce_mean
                elif reduction_str == "min":
                    reduce_fn = tf.reduce_min
                elif reduction_str == "max":
                    reduce_fn = tf.reduce_max
                else:
                    raise RuntimeError("Unknown reduction type: {}".format(reduction_str))
                if reduce_fn is not None:
                    target_image_batch = reduce_fn(target_image_batch, axis=axis)
                    predicted_image_batch = reduce_fn(predicted_image_batch, axis=axis)
                    abs_diff_image = reduce_fn(abs_diff_image, axis=axis)

        logger.info("Image summary for channel {} will be clipped to [{}, {}]".format(channel, clip_min, clip_max))
        target_image_batch = tf.clip_by_value(target_image_batch, clip_min, clip_max)
        predicted_image_batch = tf.clip_by_value(predicted_image_batch, clip_min, clip_max)
        abs_diff_image = tf.clip_by_value(abs_diff_image, clip_min, clip_max)
        if cfg.get("clip_scale", False):
            target_image_batch = 255 * (target_image_batch - clip_min) / (clip_max - clip_min)
            predicted_image_batch = 255 * (predicted_image_batch - clip_min) / (clip_max - clip_min)
            abs_diff_image = 255 * (abs_diff_image - clip_min) / (clip_max - clip_min)
            target_image_batch = tf.cast(target_image_batch, tf.uint8)
            predicted_image_batch = tf.cast(predicted_image_batch, tf.uint8)
            abs_diff_image = tf.cast(abs_diff_image, tf.uint8)

        if "tile_multiples" in cfg:
            # Add batch and channel dimension to tile multiples
            tile_multiples = [1] + cfg.tile_multiples + [1]
            target_image_batch = tf.tile(target_image_batch, tile_multiples)
            predicted_image_batch = tf.tile(predicted_image_batch, tile_multiples)
            abs_diff_image = tf.tile(abs_diff_image, tile_multiples)

        if "resize" in cfg:
            new_size = cfg.resize
            resize_method_name = cfg.get("resize_method", "nearest")
            if resize_method_name == "nearest":
                resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
            elif resize_method_name == "bilinear":
                resize_method = tf.image.ResizeMethod.BILINEAR
            elif resize_method_name == "bicubic":
                resize_method = tf.image.ResizeMethod.BICUBIC
            elif resize_method_name == "area":
                resize_method = tf.image.ResizeMethod.AREA
            else:
                raise RuntimeError("Unknown image resize method: {}".format(resize_method_name))
            target_image_batch = tf.image.resize_images(target_image_batch, new_size, method=resize_method)
            predicted_image_batch = tf.image.resize_images(predicted_image_batch, new_size, method=resize_method)
            abs_diff_image_batch = tf.image.resize_images(abs_diff_image_batch, new_size, method=resize_method)

        image_summary_count = cfg.get("image_summary_count", 16)
        image_summaries.extend([
            tf.summary.image("image/target/{}".format(channel), target_image_batch, image_summary_count),
            tf.summary.image("image/predicted/{}".format(channel), predicted_image_batch, image_summary_count),
            tf.summary.image("image/abs_diff/{}".format(channel), abs_diff_image, image_summary_count),
        ])
    return image_summaries


def make_model_hist_summary(input_batch, target_batch, weight_batch, model, prefix="model"):
    input_shape = input_batch.get_shape().as_list()
    target_shape = target_batch.get_shape().as_list()
    model_summary_dict = {}
    if len(target_shape) > 1:
        for j in range(target_shape[-1]):
            model_summary_dict["target_batch/{}".format(j)] \
                = target_batch[..., slice(j, j + 1)]
        model_summary_dict["target_batch/all"] = target_batch
    else:
        model_summary_dict["target_batch"] = target_batch
    for j in range(input_shape[-1]):
        model_summary_dict["input_batch/{}".format(j)] \
            = input_batch[..., slice(j, j + 1)]
    model_summary_dict["input_batch/all"] = input_batch
    model_summary_dict["weight_batch"] = weight_batch

    var_to_grad_dict = {var: grad for var, grad in zip(model.variables, model.gradients)}
    for name, tensor in model.summaries.items():
        model_summary_dict[name] = tensor
        # Reuse existing gradient expressions
        if tensor in var_to_grad_dict:
            grad = var_to_grad_dict[tensor]
            if grad is not None:
                model_summary_dict[name + "_grad"] = grad

    model_summary_dict = {prefix + "/" + name: value for name, value in model_summary_dict.items()}
    model_hist_summary = tf_utils.HistogramSummary(model_summary_dict)
    return model_hist_summary


def run_training_epoch(cfg, sess, epoch, dataflow, model, batch_feeder,
                   batch_size, train_op, inc_epoch, learning_rate_tf,
                   global_step_tf, var_global_norm, grad_global_norm,
                   total_batch_count, total_sample_count,
                   image_summary_op,
                   target_tensor_joined,
                   train_tf_queue_size_fetch, test_tf_queue_size_fetch,
                   train_staging_areas, test_staging_areas,
                   summary_wrapper, minibatch_summary_wrapper, model_hist_summary, summary_writer,
                   log_pearson_correlation, log_spearman_correlation,
                   run_options, run_metadata,
                   verbose):
    timer = Timer()
    total_unweighted_loss_value = 0.0
    total_unregularized_loss_value = 0.0
    total_regularization_value = 0.0
    total_loss_value = 0.0
    total_loss_min = +np.finfo(np.float32).max
    total_loss_max = -np.finfo(np.float32).max
    batch_count = 0

    summary_op = minibatch_summary_wrapper.get_summary_op()

    do_summary = cfg.io.train_summary_interval > 0 and epoch % cfg.io.train_summary_interval == 0
    if do_summary:
        var_global_norm_v = 0.0
        grad_global_norm_v = 0.0
        if verbose:
            logger.info("Generating train summary")

    do_model_summary = cfg.io.model_summary_interval > 0 and epoch % cfg.io.model_summary_interval == 0
    if do_model_summary:
        model_summary_fetched = [[] for _ in model_hist_summary.fetches]
        if verbose:
            logger.info("Generating model summary")

    if cfg.io.debug_performance:
        do_summary = False
        do_model_summary = False

    if log_pearson_correlation or log_spearman_correlation:
        outputs = np.empty((0,))
        targets = np.empty((0,))

    # Training loop for one epoch
    epoch_pbar = progressbar.get_progressbar(total=dataflow.size(),
                                             desc="Epoch {}".format(epoch),
                                             unit="sample")
    for sample_count in range(0, dataflow.size(), batch_size):
        if verbose:
            logger.info("Training batch # {}, epoch {}, sample # {}".format(batch_count, epoch, sample_count))
            logger.info("  Total batch # {}, total sample # {}".format(total_batch_count, total_sample_count))

        # Create train op list
        fetches = [train_op, global_step_tf, model.unweighted_loss, model.unregularized_loss,
                   model.regularization, model.loss, model.loss_min, model.loss_max, summary_op, global_step_tf]

        if log_pearson_correlation or log_spearman_correlation:
            output_offset = len(fetches)
            fetches += [model.output]
            target_offset = len(fetches)
            fetches += [target_tensor_joined]

        if do_summary and batch_count == 0:
            summary_offset = len(fetches)
            fetches.extend([var_global_norm, grad_global_norm])
            if image_summary_op is not None:
                fetches.append(image_summary_op)

        if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
            model_summary_offset = len(fetches)
            fetches.extend(model_hist_summary.fetches)
            model_summary_end = len(fetches)

        if cfg.io.debug_queues:
            fetches += [train_tf_queue_size_fetch, train_staging_areas[0].size]
            fetches += [test_tf_queue_size_fetch, test_staging_areas[0].size]

        fetched = \
            sess.run(fetches,
                     options=run_options,
                     run_metadata=run_metadata)
        _, global_step, unweighted_loss_v, unregularized_loss_v, regularization_v, \
        loss_v, loss_min_v, loss_max_v, minibatch_summary, global_step = fetched[:10]

        if cfg.io.debug_queues:
            # logger.info("Filename queue size: {}".format(train_pipeline.hdf5_input_pipeline.filename_queue.qsize()))
            # logger.info("Sample queue size: {}".format(train_pipeline.hdf5_input_pipeline.data_queue.qsize()))
            logger.info("Train TF queue size: {}".format(fetched[-4]))
            logger.info("Train TF staging area [0] size: {}".format(fetched[-3]))
            logger.info("Test TF queue size: {}".format(fetched[-2]))
            logger.info("Test TF staging area [0] size: {}".format(fetched[-1]))

        if do_summary and batch_count == 0:
            var_global_norm_v += fetched[summary_offset]
            grad_global_norm_v += fetched[summary_offset + 1]
            if image_summary_op is not None:
                image_summary = fetched[summary_offset + 2]
                summary_writer.add_summary(image_summary, global_step=global_step)
                summary_writer.flush()

        if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
            for i, value in enumerate(fetched[model_summary_offset:model_summary_end]):
                # Make sure we copy the model summary tensors (otherwise it might be pinned GPU memory)
                model_summary_fetched[i].append(np.array(value))

        if log_pearson_correlation or log_spearman_correlation:
            output_batch = fetched[output_offset]
            target_batch = fetched[target_offset]
            outputs = np.append(outputs, output_batch)
            targets = np.append(targets, target_batch)

        if cfg.tensorflow.create_tf_timeline:
            # Save timeline traces
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())

        if (args.verbose or cfg.io.debug_memory) and batch_count == 0:
            # Print memory usage
            max_memory_gpu_v, peak_use_memory_gpu_v = sess.run([max_memory_gpu, peak_use_memory_gpu])
            logger.info("Max memory on GPU: {} MB, peak used memory on GPU: {} MB".format(
                max_memory_gpu_v / 1024. / 1024., peak_use_memory_gpu_v / 1024. / 1024.))

        total_unweighted_loss_value += unweighted_loss_v
        total_unregularized_loss_value += unregularized_loss_v
        total_regularization_value += regularization_v
        total_loss_value += loss_v
        total_loss_min = np.minimum(loss_min_v, total_loss_min)
        total_loss_max = np.maximum(loss_max_v, total_loss_max)
        batch_count += 1
        total_batch_count += 1
        if cfg.tensorflow.use_gpu_feeders:
            batch_feeder.notify_batch_consumption()

        summary_writer.add_summary(minibatch_summary, global_step=global_step)

        epoch_pbar.update(batch_size)

    total_sample_count += dataflow.size()
    total_unweighted_loss_value /= batch_count
    total_unregularized_loss_value /= batch_count
    total_regularization_value /= batch_count
    total_loss_value /= batch_count
    epoch_pbar.close()

    if do_summary:
        if args.verbose:
            logger.info("Writing train summary")
        summary_dict = {
            "unweighted_loss": total_unweighted_loss_value,
            "unregularized_loss": total_unregularized_loss_value,
            "regularization": total_regularization_value,
            "loss": total_loss_value,
            "loss_min": total_loss_min,
            "loss_max": total_loss_max,
            "var_global_norm": var_global_norm_v,
            "grad_global_norm": grad_global_norm_v,
        }
        if log_pearson_correlation:
            pearson_correlation, _ = scipy.stats.pearsonr(outputs, targets)
            summary_dict["pearson_correlation"] = pearson_correlation
        if log_spearman_correlation:
            spearman_correlation, _ = scipy.stats.spearmanr(outputs, targets)
            summary_dict["spearman_correlation"] = spearman_correlation
        summary = summary_wrapper.create_summary(sess, summary_dict)
        try:
            summary_writer.add_summary(summary, global_step=global_step)
            summary_writer.flush()
            num_summary_errors = 0
        except Exception as exc:
            logger.error("ERROR: Exception when trying to write model summary: {}".format(exc))
            if num_summary_errors >= cfg.io.max_num_summary_errors:
                logger.fatal("Too many summary errors occured. Aborting.")
                raise

    if do_model_summary:
        # TODO: Build model summary in separate thread?
        model_summary = None
        feed_dict = {}
        with logged_time_measurement(logger, "Concatenating model summaries", log_start=True):
            for i, placeholder in enumerate(model_hist_summary.placeholders):
                feed_dict[placeholder] = np.concatenate(model_summary_fetched[i], axis=0)
            if args.verbose:
                logger.info("Building model summary")
            try:
                model_summary = sess.run([model_hist_summary.summary_op], feed_dict=feed_dict)[0]
                if args.verbose:
                    logger.info("Writing model summary")
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(model_summary, global_step=global_step)
                summary_writer.flush()
                num_summary_errors = 0
            except TFInvalidArgumentError as exc:
                logger.error("ERROR: Exception when retrieving model summary: {}".format(exc))
                traceback.print_exc()
            except Exception as exc:
                logger.error("ERROR: Exception when trying to write model summary: {}".format(exc))
                traceback.print_exc()
                if num_summary_errors >= cfg.io.max_num_summary_errors:
                    logger.fatal("Too many summary errors occured. Aborting.")
                    raise
            finally:
                del model_summary_fetched
                del feed_dict
                del model_summary

    logger.info("train result:")
    logger.info("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
        epoch, total_loss_value, total_loss_min, total_loss_max))
    if log_pearson_correlation:
        logger.info("  pearson corr: {}".format(pearson_correlation))
    if log_spearman_correlation:
        logger.info("  spearman corr: {}".format(spearman_correlation))
    epoch_time_sec = timer.restart()
    epoch_time_min = epoch_time_sec / 60.
    logger.info("train stats:")
    logger.info("  batches: {}, samples: {}, time: {} min".format(
        batch_count, sample_count, epoch_time_min))
    sample_per_sec = sample_count / float(epoch_time_sec)
    batch_per_sec = batch_count / float(epoch_time_sec)
    logger.info("  sample/s: {}, batch/s: {}".format(sample_per_sec, batch_per_sec))
    logger.info("  total batches: {}, total samples: {}".format(
        total_batch_count, total_sample_count))

    sess.run([inc_epoch])
    learning_rate = float(sess.run([learning_rate_tf])[0])
    logger.info("Current learning rate: {:e}".format(learning_rate))

    return total_batch_count, total_sample_count


def run_validation_epoch(cfg, sess, epoch, dataflow, model, batch_feeder,
                   batch_size, prefetch_op,
                   global_step_tf,
                   image_summary_op,
                   target_tensor_joined,
                   train_tf_queue_size_fetch, test_tf_queue_size_fetch,
                   train_staging_areas, test_staging_areas,
                   summary_wrapper, model_hist_summary, summary_writer,
                   log_pearson_correlation, log_spearman_correlation, verbose):
    logger.info("Performing validation run")
    timer = Timer()
    total_unweighted_loss_value = 0.0
    total_unregularized_loss_value = 0.0
    total_regularization_value = 0.0
    total_loss_value = 0.0
    total_loss_min = +np.finfo(np.float32).max
    total_loss_max = -np.finfo(np.float32).max
    batch_count = 0

    do_model_summary = cfg.io.test_model_summary_interval > 0 and epoch % cfg.io.test_model_summary_interval == 0
    if do_model_summary:
        model_summary_fetched = [[] for _ in model_hist_summary.fetches]
        if verbose:
            logger.info("Generating test model summary")

    if log_pearson_correlation or log_spearman_correlation:
        outputs = np.empty((0,))
        targets = np.empty((0,))
    epoch_pbar = progressbar.get_progressbar(total=dataflow.size(),
                                             desc="Epoch {}".format(epoch),
                                             unit="sample")
    for sample_count in range(0, dataflow.size(), batch_size):
        fetches = [prefetch_op, model.unweighted_loss,
                   model.unregularized_loss, model.regularization,
                   model.loss, model.loss_min, model.loss_max]
        if image_summary_op is not None and batch_count == 0:
            image_summary_offset = len(fetches)
            fetches += [image_summary_op, global_step_tf]
        if log_pearson_correlation or log_spearman_correlation:
            output_offset = len(fetches)
            fetches += [model.output]
            target_offset = len(fetches)
            fetches += [target_tensor_joined]

        if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
            model_summary_offset = len(fetches)
            fetches.extend(model_hist_summary.fetches)
            model_summary_end = len(fetches)

        fetched = sess.run(fetches)

        _, unweighted_loss_v, unregularized_loss_v, regularization_v, loss_v, loss_min_v, loss_max_v = fetched[:7]

        if image_summary_op is not None and batch_count == 0:
            image_summary, global_step = fetched[image_summary_offset:(image_summary_offset + 2)]
            summary_writer.add_summary(image_summary, global_step=global_step)
            summary_writer.flush()
        if log_pearson_correlation or log_spearman_correlation:
            output_batch = fetched[output_offset]
            target_batch = fetched[target_offset]
            outputs = np.append(outputs, output_batch)
            targets = np.append(targets, target_batch)

        if cfg.io.debug_queues:
            train_tf_queue_size, train_staging_size, test_tf_queue_size, test_staging_size = \
            sess.run([train_tf_queue_size_fetch,
                      train_staging_areas[0].size,
                      test_tf_queue_size_fetch,
                      test_staging_areas[0].size])
            # logger.info("Filename queue size: {}".format(train_pipeline.hdf5_input_pipeline.filename_queue.qsize()))
            # logger.info("Sample queue size: {}".format(train_pipeline.hdf5_input_pipeline.data_queue.qsize()))
            logger.info("Train TF queue size: {}".format(train_tf_queue_size))
            logger.info("Train TF staging area [0] size: {}".format(train_staging_size))
            logger.info("Test TF queue size: {}".format(test_tf_queue_size))
            logger.info("Test TF staging area [0] size: {}".format(test_staging_size))

        if do_model_summary and batch_count < cfg.io.model_summary_num_batches:
            for i, value in enumerate(fetched[model_summary_offset:model_summary_end]):
                # Make sure we copy the model summary tensors (otherwise it might be pinned GPU memory)
                model_summary_fetched[i].append(np.array(value))

        total_unweighted_loss_value += unweighted_loss_v
        total_unregularized_loss_value += unregularized_loss_v
        total_regularization_value += regularization_v
        total_loss_value += loss_v
        total_loss_min = np.minimum(loss_min_v, total_loss_min)
        total_loss_max = np.maximum(loss_max_v, total_loss_max)
        batch_count += 1
        if cfg.tensorflow.use_gpu_feeders:
            batch_feeder.notify_batch_consumption()
        epoch_pbar.update(batch_size)
    total_unweighted_loss_value /= batch_count
    total_unregularized_loss_value /= batch_count
    total_regularization_value /= batch_count
    total_loss_value /= batch_count
    epoch_pbar.close()

    summary_dict = {
        "unweighted_loss": total_unweighted_loss_value,
        "unregularized_loss": total_unregularized_loss_value,
        "regularization": total_regularization_value,
        "loss": total_loss_value,
        "loss_min": total_loss_min,
        "loss_max": total_loss_max,
    }
    if log_pearson_correlation:
        pearson_correlation, _ = scipy.stats.pearsonr(outputs, targets)
        summary_dict["pearson_correlation"] = pearson_correlation
    if log_spearman_correlation:
        spearman_correlation, _ = scipy.stats.spearmanr(outputs, targets)
        summary_dict["spearman_correlation"] = spearman_correlation
    summary = summary_wrapper.create_summary(sess, summary_dict)
    global_step = int(sess.run([global_step_tf])[0])
    summary_writer.add_summary(summary, global_step=global_step)
    summary_writer.flush()

    if do_model_summary:
        model_summary = None
        feed_dict = {}
        with logged_time_measurement(logger, "Concatenating test model summaries", log_start=True):
            for i, placeholder in enumerate(model_hist_summary.placeholders):
                feed_dict[placeholder] = np.concatenate(model_summary_fetched[i], axis=0)
            if args.verbose:
                logger.info("Building test model summary")
            try:
                model_summary = sess.run([model_hist_summary.summary_op], feed_dict=feed_dict)[0]
                if args.verbose:
                    logger.info("Writing test model summary")
                global_step = int(sess.run([global_step_tf])[0])
                summary_writer.add_summary(model_summary, global_step=global_step)
                summary_writer.flush()
                num_summary_errors = 0
            except TFInvalidArgumentError as exc:
                logger.error("ERROR: Exception when retrieving test model summary: {}".format(exc))
                traceback.print_exc()
            except Exception as exc:
                logger.error("ERROR: Exception when trying to write test model summary: {}".format(exc))
                traceback.print_exc()
                if num_summary_errors >= cfg.io.max_num_summary_errors:
                    logger.fatal("Too many summary errors occured. Aborting.")
                    raise
            finally:
                del model_summary_fetched
                del feed_dict
                del model_summary

    logger.info("------------")
    logger.info("test result:")
    logger.info("  epoch: {}, loss: {}, min loss: {}, max loss: {}".format(
        epoch, total_loss_value, total_loss_min, total_loss_max))
    if log_pearson_correlation:
        logger.info("  pearson corr: {}".format(pearson_correlation))
    if log_spearman_correlation:
        logger.info("  spearman corr: {}".format(spearman_correlation))
    sample_count = batch_count * batch_size
    epoch_time_sec = timer.restart()
    epoch_time_min = epoch_time_sec / 60.
    logger.info("test stats:")
    logger.info("  batches: {}, samples: {}, time: {} min".format(
        batch_count, sample_count, epoch_time_min))
    sample_per_sec = sample_count / float(epoch_time_sec)
    batch_per_sec = batch_count / float(epoch_time_sec)
    logger.info("  sample/s: {}, batch/s: {}".format(sample_per_sec, batch_per_sec))
    logger.info("------------")


def save_model(sess, saver, model_dir, model_name, global_step_tf, max_trials, retry_save_wait_time=5, verbose=False):
    saved = False
    trial = 0
    model_filename = os.path.join(model_dir, model_name)
    while not saved and trial < max_trials:
        try:
            trial += 1
            timer = Timer()
            if verbose:
                if trial > 1:
                    logger.info("Saving model to {}. Trial {}.".format(model_filename, trial))
                else:
                    logger.info("Saving model to {}".format(model_filename))
            filename = saver.save(sess, model_filename, global_step=global_step_tf)
            save_time = timer.restart()
            saved = True
            if verbose:
                logger.info("Saving took {} s".format(save_time))
                logger.info("Saved model to file: {}".format(filename))
            latest_checkpoint = tf.train.latest_checkpoint(model_dir)
            assert(latest_checkpoint is not None)
            latest_checkpoint_basename = os.path.basename(latest_checkpoint)
            model_ckpt_name = os.path.basename(filename)
            assert(latest_checkpoint_basename == model_ckpt_name)
            assert(os.path.isfile(filename + ".index"))
            assert(os.path.isfile(filename + ".meta"))
        except Exception as err:
            logger.error("ERROR: Exception when trying to save model: {}".format(err))
            traceback.print_exc()
            if trial < max_trials:
                if verbose:
                    logger.error("Retrying to save model in {} s...".format(retry_save_wait_time))
                time.sleep(retry_save_wait_time)
            else:
                raise RuntimeError("Unable to save model after {} trials".format(max_trials))


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

    cfg.io.retry_save_wait_time = 3
    cfg.io.max_num_summary_errors = 3
    if cfg.io.model_summary_num_batches <= 0:
        cfg.io.model_summary_num_batches = np.iinfo(np.int32).max

    # Learning parameters
    num_epochs = cfg.training.num_epochs
    batch_size = cfg.training.batch_size
    test_batch_size = cfg.training.test_batch_size
    if test_batch_size <= 0:
        test_batch_size = batch_size

    use_sample_weights = False
    if cfg.data.get("weight", None) is not None and not cfg.data.weight.get("uniform", False):
        use_sample_weights = True

        def normalize_bin_weights(hist, weights):
            alpha = np.sum(hist) / np.sum(hist * weights)
            return alpha * weights

        def compute_bin_weights(hist, power=2, truncation=10.):
            total_count = np.sum(hist)
            weights = np.power(total_count / np.asarray(hist, dtype=np.float32), power)
            weights[np.logical_not(np.isfinite(weights))] = np.max(weights[np.isfinite(weights)])
            weights = normalize_bin_weights(hist, weights)
            weights[weights > truncation] = truncation
            weights = normalize_bin_weights(hist, weights)
            return weights

        def compute_sample_weight(sample, weights, bin_edges):
            sample = np.minimum(sample, bin_edges[-1])
            bin = np.argmin((sample > bin_edges).astype(np.int), axis=-1) - 1
            bin = np.maximum(bin, 0)
            return weights[bin]

        def compute_sample_hist(data_path, data_cfg, num_bins=20):
            histogram_dataflow = input_pipeline.InputAndTargetDataFlow(data_path, cfg.data, repeats=1, verbose=True)
            histogram_dataflow.reset_state()
            target_values = np.zeros((histogram_dataflow.size(), 1))
            pbar = progressbar.get_progressbcar(total=len(target_values), miniters=1)
            min_v = np.finfo(np.float32).max
            max_v = -np.finfo(np.float32).max
            counter = 0
            for i, dp in enumerate(histogram_dataflow.get_batch_dataflow().get_data()):
                target_batch = dp[1]
                target_values[counter:counter+len(target_batch)] = target_batch
                min_v = min([min_v, np.min(target_batch)])
                max_v = max([max_v, np.max(target_batch)])
                counter += len(target_batch)
                pbar.update(len(target_batch))
            pbar.close()
            del histogram_dataflow
            hist, bin_edges = np.histogram(target_values, num_bins)
            logger.info("Minimum: {}".format(min_v))
            logger.info("Maximum: {}".format(max_v))
            return hist, bin_edges

        weight_power = cfg.data.weight.get("power", 1.3)
        num_bins = cfg.data.weight.get("num_bins", 20)
        weight_truncation = cfg.data.weight.get("truncation", 5)

        logger.info("Computing train dataset histogram")
        train_sample_hist, train_sample_hist_bin_edges = compute_sample_hist(cfg.data.train_path, cfg.data, num_bins)
        logger.info("Train sample histogram: {}".format(train_sample_hist))
        logger.info("Train sample bin edges: {}".format(train_sample_hist_bin_edges))
        train_sample_weights = compute_bin_weights(train_sample_hist, weight_power, weight_truncation)
        logger.info("Train sample weights: {}".format(train_sample_weights))

        logger.info("Computing test dataset histogram")
        test_sample_hist, test_sample_hist_bin_edges = compute_sample_hist(cfg.data.test_path, cfg.data, num_bins)
        logger.info("Test sample histogram: {}".format(test_sample_hist))
        logger.info("Test sample bin edges: {}".format(test_sample_hist_bin_edges))
        test_sample_weights = compute_bin_weights(test_sample_hist, weight_power, weight_truncation)
        logger.info("Test sample weights: {}".format(test_sample_weights))

    train_dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.train_path, cfg.data,
                                                           random_start_position=True, verbose=True)
    test_dataflow = input_pipeline.InputAndTargetDataFlow(cfg.data.test_path, cfg.data, verbose=True,
                                                          override_data_stats=train_dataflow.get_data_stats())

    logger.info("# samples in train dataset: {}".format(train_dataflow.size()))
    logger.info("# samples in test dataset: {}".format(test_dataflow.size()))

    logger.info("Input and target shapes:")
    train_dataflow.reset_state()
    first_sample = next(train_dataflow.get_data())
    tensor_shapes = [tensor.shape for tensor in first_sample] + [(1,)]
    tensor_dtypes = [tensor.dtype for tensor in first_sample] + [np.float32]
    logger.info("  Shape of input: {}".format(first_sample[0].shape))
    logger.info("  Type of input: {}".format(first_sample[0].dtype))
    logger.info("  Shape of target: {}".format(first_sample[1].shape))
    logger.info("  Type of target: {}".format(first_sample[1].dtype))
    logger.info("  Shape of weights: {}".format(tensor_shapes[2]))
    logger.info("  Type of weights: {}".format(tensor_dtypes[2]))

    # if args.verbose:
    #     input_pipeline.print_data_stats(train_filenames[0], input_and_target_retriever)

    # Create tensorflow session
    logger.info("Creating tensorflow session")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.tensorflow.gpu_memory_fraction)
    tf_config = tf.ConfigProto(gpu_options=gpu_options)
    tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
    tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
    tf_config.log_device_placement = cfg.tensorflow.log_device_placement
    with tf.Session(config=tf_config) as sess:
        # Setup TF step and epoch counters
        global_step_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='global_step')
        epoch_tf = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False, name='epoch')
        inc_epoch = epoch_tf.assign_add(tf.constant(1, dtype=tf.int64))

        coord = tf.train.Coordinator()

        # def signal_handler(signal, frame):
        #     sess.close()
        # signal.signal(signal.SIGINT, signal.default_int_handler)

        if use_sample_weights:
            def add_weights_to_datapoint(dp):
                target_batch = dp[1]
                weight_batch = compute_sample_weight(target_batch, train_sample_hist, train_sample_hist_bin_edges)
                weight_batch = weight_batch[..., np.newaxis]
                return [dp[0], target_batch, weight_batch]
        else:
            def add_weights_to_datapoint(dp):
                target_batch = dp[1]
                weight_batch = np.ones((target_batch.shape[0], 1))
                return [dp[0], target_batch, weight_batch]

        # TODO: Cleanup weight handling
        import tensorpack.dataflow

        with tf.device("/cpu:0"):
            # Important: Stats for normalizing should be the same for both train and test dataset
            sample_stats = train_dataflow.get_sample_stats()
            train_batch_dataflow = train_dataflow.get_batch_dataflow()
            train_batch_dataflow = tensorpack.dataflow.MapData(train_batch_dataflow, add_weights_to_datapoint)
            train_pipeline = input_pipeline.TFDataFlowPipeline(
                train_batch_dataflow, tensor_shapes, tensor_dtypes, sess, coord, batch_size,
                cfg.tensorflow, is_training=True, sample_stats=sample_stats, is_batch_dataflow=True)
            test_batch_dataflow = test_dataflow.get_batch_dataflow()
            test_batch_dataflow = tensorpack.dataflow.MapData(test_batch_dataflow, add_weights_to_datapoint)
            test_pipeline = input_pipeline.TFDataFlowPipeline(
                test_batch_dataflow, tensor_shapes, tensor_dtypes, sess, coord, test_batch_size,
                cfg.tensorflow, is_training=False, sample_stats=sample_stats, is_batch_dataflow=True)

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
        filtered_device_names = [x for x in device_names if x in existing_device_names]
        if len(filtered_device_names) < len(device_names):
            logger.warn("WARNING: Not all desired devices are available.")
            logger.info("Available devices:")
            for device_name in existing_device_names:
                logger.info("  {}".format(device_name))
            logger.info("Filtered devices:")
            for device_name in filtered_device_names:
                logger.info("  {}".format(device_name))
            if cfg.tensorflow.strict_devices:
                import sys
                sys.exit(1)
            logger.info("Continuing with filtered devices")
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

        batch_size_per_device = batch_size // len(device_names)
        test_batch_size_per_device = test_batch_size // len(device_names)
        if not batch_size % batch_size_per_device == 0:
            logger.info("Batch size has to be equally divideable by number of devices")
        if not test_batch_size % test_batch_size_per_device == 0:
            logger.info("Test batch size has to be equally divideable by number of devices")

        logger.info("Used devices:")
        for device_name in device_names:
            logger.info("  {}".format(device_name))
        if use_multi_gpu:
            logger.info("Parameter server device: {}".format(ps_device_name))
        logger.info("Total train batch size: {}, per device batch size: {}".format(batch_size, batch_size_per_device))
        logger.info("Total test batch size: {}, per device batch size: {}".format(test_batch_size, test_batch_size_per_device))

        # Create model

        train_staging_areas = []
        train_models = []
        test_staging_areas = []
        test_models = []

        train_input_tensors_splits = list(zip(*[
            tf.split(tensor, len(device_names)) for tensor in train_pipeline.tensors_batch
        ]))
        test_input_tensors_splits = list(zip(*[
            tf.split(tensor, len(device_names)) for tensor in test_pipeline.tensors_batch
        ]))
        for i, device_name in enumerate(device_names):
            with tf.device(device_name):
                # input_batch = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name="in_input")
                reuse = i > 0
                with tf.variable_scope("model", reuse=reuse):
                    staging_area = data_provider.TFStagingArea(train_input_tensors_splits[i], device_name)
                    train_staging_areas.append(staging_area)
                    model = models.Model(model_config,
                                         staging_area.tensors[0],
                                         staging_area.tensors[1],
                                         staging_area.tensors[2],
                                         is_training=True,
                                         variables_on_cpu=variables_on_cpu,
                                         verbose=max(args.verbose, args.verbose_model))
                    train_models.append(model)
                # Generate output and loss function for testing (no dropout)
                with tf.variable_scope("model", reuse=True):
                    staging_area = data_provider.TFStagingArea(test_input_tensors_splits[i], device_name)
                    test_staging_areas.append(staging_area)
                    model = models.Model(model_config,
                                         staging_area.tensors[0],
                                         staging_area.tensors[1],
                                         staging_area.tensors[2],
                                         is_training=False,
                                         variables_on_cpu=variables_on_cpu,
                                         verbose=max(args.verbose, args.verbose_model))
                    test_models.append(model)
        train_input_tensor_joined = tf.concat([staging_area.tensors[0] for staging_area in train_staging_areas], axis=0)
        train_target_tensor_joined = tf.concat([staging_area.tensors[1] for staging_area in train_staging_areas], axis=0)
        train_weight_tensor_joined = tf.concat([staging_area.tensors[2] for staging_area in train_staging_areas], axis=0)
        test_input_tensor_joined = tf.concat([staging_area.tensors[0] for staging_area in test_staging_areas], axis=0)
        test_target_tensor_joined = tf.concat([staging_area.tensors[1] for staging_area in test_staging_areas], axis=0)
        test_weight_tensor_joined = tf.concat([staging_area.tensors[2] for staging_area in test_staging_areas], axis=0)

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
            if cfg.training.learning_rate_decay_mode == "exponential":
                learning_rate_tf = tf.train.exponential_decay(cfg.training.initial_learning_rate,
                                                              epoch_tf,
                                                              cfg.training.learning_rate_decay_epochs,
                                                              cfg.training.learning_rate_decay_rate,
                                                              cfg.training.learning_rate_decay_staircase)
            elif cfg.training.learning_rate_decay_mode == "inverse":
                learning_rate_tf = tf.train.inverse_time_decay(cfg.training.initial_learning_rate,
                                                               epoch_tf,
                                                               cfg.training.learning_rate_decay_epochs,
                                                               cfg.training.learning_rate_decay_rate,
                                                               cfg.training.learning_rate_decay_staircase)
            else:
                raise ValueError("Unknown learning rate decay mode: {}".format(cfg.training.learning_rate_decay_mode))

            opt = optimizer_class(learning_rate_tf)

            # Get variables and gradients
            variables = train_model.variables
            gradients = train_model.gradients
            gradients, _ = tf.clip_by_global_norm(gradients, cfg.training.max_grad_global_norm)
            gradients_and_variables = list(zip(gradients, variables))
            # gradients_and_variables = opt.compute_gradients(train_model.loss, variables)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(gradients_and_variables, global_step=global_step_tf)

            var_global_norm = tf.global_norm(variables)
            grad_global_norm = tf.global_norm(gradients)

        log_pearson_correlation = cfg.io.get("pearson_correlation", True)
        log_spearman_correlation = cfg.io.get("spearman_correlation", True)

        with tf.device(tf_utils.cpu_device_name()):
            # Tensorboard summaries
            with tf.name_scope('training_minibatch'):
                minibatch_summary_wrapper = tf_utils.ScalarSummaryWrapper()
                minibatch_summary_wrapper.append_tensor("grad_global_norm", grad_global_norm)
                minibatch_summary_wrapper.append_tensor("loss", train_model.loss)
                minibatch_summary_wrapper.append_tensor("unweighted_loss", train_model.unweighted_loss)
                minibatch_summary_wrapper.append_tensor("unregularized_loss", train_model.unregularized_loss)
                minibatch_summary_wrapper.append_tensor("regularization", train_model.regularization)
                minibatch_summary_wrapper.append_tensor("loss_min", train_model.loss_min)
                minibatch_summary_wrapper.append_tensor("loss_max", train_model.loss_max)

            with tf.name_scope('training'):
                train_summary_wrapper = tf_utils.ScalarSummaryWrapper()
                models.add_model_loss_summaries(train_summary_wrapper)
                train_summary_wrapper.append_with_placeholder("grad_global_norm", tf.float32)
                train_summary_wrapper.append_with_placeholder("var_global_norm", tf.float32)
                if log_pearson_correlation:
                    train_summary_wrapper.append_with_placeholder("pearson_correlation", tf.float32)
                if log_spearman_correlation:
                    train_summary_wrapper.append_with_placeholder("spearman_correlation", tf.float32)

                image_summaries = []
                if "input_target_image_summary" in cfg.io:
                    summaries = make_input_target_image_summaries(
                        cfg.io["input_target_image_summary"], train_pipeline.unnormalized_target_batch_tf, train_model.output)
                    image_summaries.extend(summaries)
                if "input_image_summary" in cfg.io:
                    summaries = make_image_summaries(
                        cfg.io["input_image_summary"], train_pipeline.unnormalized_input_batch_tf, "input")
                    image_summaries.extend(summaries)
                if "target_image_summary" in cfg.io:
                    summaries = make_image_summaries(
                        cfg.io["target_image_summary"], train_pipeline.unnormalized_target_batch_tf, "target")
                    image_summaries.extend(summaries)
                if len(image_summaries) > 0:
                    train_image_summary_op = tf.summary.merge(image_summaries)
                else:
                    train_image_summary_op = None

            with tf.name_scope('testing'):
                test_summary_wrapper = tf_utils.ScalarSummaryWrapper()
                models.add_model_loss_summaries(test_summary_wrapper)
                if log_pearson_correlation:
                    test_summary_wrapper.append_with_placeholder("pearson_correlation", tf.float32)
                if log_spearman_correlation:
                    test_summary_wrapper.append_with_placeholder("spearman_correlation", tf.float32)

                image_summaries = []
                if "input_target_image_summary" in cfg.io:
                    summaries = make_input_target_image_summaries(
                        cfg.io["input_target_image_summary"], test_pipeline.unnormalized_target_batch_tf, test_model.output)
                    image_summaries.extend(summaries)
                if "input_image_summary" in cfg.io:
                    summaries = make_image_summaries(
                        cfg.io["input_image_summary"], test_pipeline.unnormalized_input_batch_tf, "input")
                    image_summaries.extend(summaries)
                if "target_image_summary" in cfg.io:
                    summaries = make_image_summaries(
                        cfg.io["target_image_summary"], test_pipeline.unnormalized_target_batch_tf, "target")
                    image_summaries.extend(summaries)
                if len(image_summaries) > 0:
                    test_image_summary_op = tf.summary.merge(image_summaries)
                else:
                    test_image_summary_op = None

            with tf.name_scope('progress'):
                train_summary_wrapper.append_tensor("epoch", epoch_tf)
                train_summary_wrapper.append_tensor("learning_rate", learning_rate_tf)
            if cfg.io.debug_summary:
                with tf.name_scope('debug'):
                    train_summary_wrapper.append_tensor("train_input_queue/size", train_pipeline.tf_queue.size())
                    for i, sa in enumerate(train_staging_areas):
                        train_summary_wrapper.append_tensor("train_staging_area/{}/size".format(i), sa.size)
                    test_summary_wrapper.append_tensor("test_input_queue/size", test_pipeline.tf_queue.size())
                    for i, sa in enumerate(test_staging_areas):
                        test_summary_wrapper.append_tensor("test_staging_area/{}/size".format(i), sa.size)

        if args.verbose:
            logger.info("Tensorflow variables:")
            for var in tf.global_variables():
                logger.info("  {}: {}".format(var.name, var.shape))
        logger.info("Model variables:")
        for grad, var in gradients_and_variables:
            logger.info("  {}: {}".format(var.name, var.shape))

        with tf.device(tf_utils.cpu_device_name()):
            # Model histogram summaries
            if cfg.io.model_summary_interval > 0:
                with tf.device("/cpu:0"):
                    train_model_hist_summary = make_model_hist_summary(
                        train_pipeline.unnormalized_input_batch_tf,
                        train_pipeline.unnormalized_target_batch_tf,
                        train_pipeline.weight_batch_tf,
                        train_model,
                        "train")
            else:
                train_model_hist_summary = None
            if cfg.io.test_model_summary_interval > 0:
                with tf.device("/cpu:0"):
                    test_model_hist_summary = make_model_hist_summary(
                        test_pipeline.unnormalized_input_batch_tf,
                        test_pipeline.unnormalized_target_batch_tf,
                        test_pipeline.weight_batch_tf,
                        test_model,
                        "test")
            else:
                test_model_hist_summary = None

        saver = tf.train.Saver(max_to_keep=cfg.io.keep_n_last_checkpoints,
                               keep_checkpoint_every_n_hours=cfg.io.keep_checkpoint_every_n_hours,
                               save_relative_paths=True)

        models.report_model_size(train_model, logger)

        # Initialize tensorflow session
        if cfg.tensorflow.create_tf_timeline:
            run_metadata = tf.RunMetadata()
        else:
            run_metadata = None
        init = tf.global_variables_initializer()
        sess.run(init)

        if args.restore or args.try_restore:
            checkpoint_path = None
            # Try to restore model
            if args.checkpoint is None:
                logger.info("Reading latest checkpoint from {}".format(args.model_dir))
                ckpt = tf.train.get_checkpoint_state(args.model_dir)
                if ckpt is None:
                    if args.restore:
                        raise IOError("No previous checkpoint found at {}".format(args.model_dir))
                    else:
                        logger.warn("WARNING: Could not find previous checkpoint. Starting from scratch.")
                else:
                    logger.info('Found previous checkpoint... restoring')
                    checkpoint_path = ckpt.model_checkpoint_path
                    # saver.recover_last_checkpoints(args.model_dir)
                    saver.restore(sess, checkpoint_path)
                    logger.info("Successfully restored model")
            else:
                checkpoint_path = os.path.join(args.model_dir, args.checkpoint)
            if checkpoint_path is not None:
                logger.info("Trying to restore model from checkpoint {}".format(checkpoint_path))
                saver.restore(sess, checkpoint_path)
                logger.info("Successfully restored model")

        try:
            train_batch_feeder = None
            test_batch_feeder = None
            custom_threads = []

            train_pipeline.start()
            test_pipeline.start()
            custom_threads.extend(train_pipeline.threads)
            custom_threads.extend(test_pipeline.threads)

            logger.info("Listing child processes")
            import os
            import signal
            import psutil
            current_process = psutil.Process()
            direct_children = current_process.children(recursive=False)
            children = current_process.children(recursive=True)
            logger.info("Num children: {} (direct: {})".format(len(children), len(direct_children)))
            for child in children:
                logger.info('Child PID is {}'.format(child.pid))

            # Start data provider threads
            custom_threads.extend(tf.train.start_queue_runners(sess=sess))

            # Tensorboard summary writer
            log_dir = args.log_dir if args.log_dir is not None else args.model_dir
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

            # Create timelines for profiling?
            if cfg.tensorflow.create_tf_timeline:
                train_options = tf.RunOptions(timeout_in_ms=100 * 1000,
                                              trace_level=tf.RunOptions.FULL_TRACE)
            else:
                train_options = None

            # Get prefetch ops and make sure the GPU pipeline is filled with a mini-batch
            train_prefetch_ops = [staging_area.put_op for staging_area in train_staging_areas]
            test_prefetch_ops = [staging_area.put_op for staging_area in test_staging_areas]
            train_prefetch_op = tf.group(*train_prefetch_ops)
            test_prefetch_op = tf.group(*test_prefetch_ops)

            if cfg.tensorflow.use_gpu_feeders:
                train_batch_feeder = data_provider.TFBatchFeeder(
                    sess, coord, train_prefetch_ops, cfg.tensorflow.staging_prefetch_group_size)
                test_batch_feeder = data_provider.TFBatchFeeder(
                    sess, coord, test_prefetch_ops, cfg.tensorflow.staging_prefetch_group_size)
                custom_threads.extend([train_batch_feeder.thread, test_batch_feeder.thread])
                train_prefetch_op = tf.no_op()
                test_prefetch_op = tf.no_op()
            # Prefetch next batch to GPU during training step
            train_op = tf.group(train_op, train_prefetch_op)

            # Report memory usage for first device. Should be same for all.
            with tf.device(device_names[0]):
                max_memory_gpu = tf_memory_stats.BytesLimit()
                peak_use_memory_gpu = tf_memory_stats.MaxBytesInUse()

            train_summary_wrapper.finalize()
            minibatch_summary_wrapper.finalize()
            test_summary_wrapper.finalize()

            train_tf_queue_size_fetch = train_pipeline.tf_queue.size()
            test_tf_queue_size_fetch = test_pipeline.tf_queue.size()

            model_check_numerics_ops = []
            for grad, var in gradients_and_variables:
                model_check_numerics_ops.append(
                    tf.check_numerics(var, "Numeric check for tensor {} failed".format(var.name)))
            model_check_numerics_op = tf.group(*model_check_numerics_ops)

            def model_check_numerics():
                import traceback
                try:
                    sess.run(model_check_numerics_op)
                except TFInvalidArgumentError as exc:
                    logger.exception("Model check numerics failed")

            if cfg.io.debug_numerics:
                logger.info("Creating ops to debug tensor numerics")
                train_check_numerics_ops = []
                train_check_numerics_ops.extend(model_check_numerics_ops)
                train_check_numerics_ops.append(tf.check_numerics(train_pipeline.input_batch_tf,
                                                                  "Numeric check for input tensor failed"))
                train_check_numerics_ops.append(tf.check_numerics(train_pipeline.target_batch_tf,
                                                                  "Numeric check for target tensor failed"))
                train_check_numerics_ops.append(tf.check_numerics(train_pipeline.unnormalized_input_batch_tf,
                                                                  "Numeric check for unnormalized input tensor failed"))
                train_check_numerics_ops.append(tf.check_numerics(train_pipeline.unnormalized_target_batch_tf,
                                                                  "Numeric check for unnormalized target tensor failed"))
                logger.info("Created {} operations to check numerics".format(len(train_check_numerics_ops)))
                train_check_numerics_op = tf.group(*train_check_numerics_ops)
                train_op = tf.group(train_op, train_check_numerics_op)

            sess.graph.finalize()

            with logged_time_measurement(logger, "Waiting for tensorflow train queue to fill up", log_start=True):
                pbar = progressbar.get_progressbar(total=cfg.tensorflow.sample_queue_min_after_dequeue + batch_size,
                                                   unit="batch",
                                                   miniters=1)
                tf_queue_size = 0
                while tf_queue_size < cfg.tensorflow.sample_queue_min_after_dequeue + batch_size:
                    tf_queue_size = sess.run(train_tf_queue_size_fetch)
                    pbar.n = tf_queue_size
                    pbar.update()
                    time.sleep(0.1)
                pbar.close()

            if cfg.tensorflow.use_gpu_feeders:
                logger.info("Starting GPU feeders")
                train_batch_feeder.start()
                test_batch_feeder.start()
            else:
                with logged_time_measurement(logger, "Prefetching batches to GPU", log_start=True):
                    pbar = progressbar.get_progressbar(total=cfg.tensorflow.sample_queue_min_after_dequeue + batch_size,
                                                       unit="batch",
                                                       miniters=1)
                    for i in range(cfg.tensorflow.staging_prefetch_group_size):
                        sess.run([train_prefetch_op, test_prefetch_op])
                        pbar.update()
                    pbar.close()

            logger.info("Starting training")

            initial_epoch = int(sess.run([epoch_tf])[0])
            total_batch_count = 0
            total_sample_count = 0
            # Training loop for all epochs
            overall_pbar = progressbar.get_progressbar(total=num_epochs,
                                                       desc="Overall progress", unit="epoch")
            overall_pbar.n = initial_epoch
            for epoch in range(initial_epoch, num_epochs):
                if coord.should_stop():
                    break

                if cfg.io.validation_interval > 0 and cfg.io.initial_validation_run and epoch == 0:
                    # Perform validation
                    run_validation_epoch(
                        cfg, sess, epoch, test_dataflow, test_model, test_batch_feeder,
                        test_batch_size, test_prefetch_op,
                        global_step_tf,
                        test_image_summary_op,
                        test_target_tensor_joined,
                        train_tf_queue_size_fetch, test_tf_queue_size_fetch,
                        train_staging_areas, test_staging_areas,
                        test_summary_wrapper, test_model_hist_summary, summary_writer,
                        log_pearson_correlation, log_spearman_correlation,
                        args.verbose)

                total_batch_count, total_sample_count = run_training_epoch(
                    cfg, sess, epoch, train_dataflow, train_model, train_batch_feeder,
                    batch_size, train_op, inc_epoch, learning_rate_tf,
                    global_step_tf, var_global_norm, grad_global_norm,
                    total_batch_count, total_sample_count,
                    train_image_summary_op,
                    train_target_tensor_joined,
                    train_tf_queue_size_fetch, test_tf_queue_size_fetch,
                    train_staging_areas, test_staging_areas,
                    train_summary_wrapper, minibatch_summary_wrapper, train_model_hist_summary, summary_writer,
                    log_pearson_correlation, log_spearman_correlation,
                    train_options, run_metadata,
                    args.verbose)

                if cfg.io.validation_interval > 0 \
                        and epoch % cfg.io.validation_interval == 0:
                    # Perform validation
                    run_validation_epoch(
                        cfg, sess, epoch, test_dataflow, test_model, test_batch_feeder,
                        test_batch_size, test_prefetch_op,
                        global_step_tf,
                        test_image_summary_op,
                        test_target_tensor_joined,
                        train_tf_queue_size_fetch, test_tf_queue_size_fetch,
                        train_staging_areas, test_staging_areas,
                        test_summary_wrapper, test_model_hist_summary, summary_writer,
                        log_pearson_correlation, log_spearman_correlation,
                        args.verbose)

                if epoch > 0 and epoch % cfg.io.checkpoint_interval == 0:
                    logger.info("Saving model at epoch {}".format(epoch))
                    model_check_numerics()
                    save_model(sess, saver, args.model_dir, "model.ckpt", global_step_tf,
                               cfg.io.max_checkpoint_save_trials, cfg.io.retry_save_wait_time, verbose=True)

                # Heartbeat signal for Philly cluster
                progress = 100 * float(epoch - initial_epoch) / (num_epochs - initial_epoch)
                logger.info("PROGRESS: {:05.2f}%".format(progress))

                # Inform about config files to better distinguish long running processes
                if args.model_config is None:
                    logger.info("Finished epoch for config={}".format(args.config))
                else:
                    logger.info("Finished epoch for config={}, model_config={}".format(args.config, args.model_config))

                # Update overall progress after each epoch
                overall_pbar.update()

            overall_pbar.close()

            logger.info("Saving final model")
            model_check_numerics()
            save_model(sess, saver, args.model_dir, "model.ckpt", global_step_tf,
                       cfg.io.max_checkpoint_save_trials, cfg.io.retry_save_wait_time, verbose=True)

        except Exception as exc:
            logger.info("Exception in training loop: {}".format(exc))
            traceback.print_exc()
            coord.request_stop(exc)
            raise exc
        finally:
            logger.info("Requesting stop")
            coord.request_stop()
            train_pipeline.stop()
            test_pipeline.stop()
            if cfg.tensorflow.use_gpu_feeders:
                if train_batch_feeder is not None:
                    train_batch_feeder.abort()
                if test_batch_feeder is not None:
                    test_batch_feeder.abort()
            coord.join(custom_threads, stop_grace_period_secs=(2 * cfg.io.timeout))
            sess.close()


if __name__ == '__main__':
    np.set_printoptions(threshold=50)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Set verbosity level.')
    parser.add_argument('--verbose-model', action='count',
                        default=0, help='Set verbosity level for model generation.')
    parser.add_argument('--model-dir', required=True, help='Model directory.')
    parser.add_argument('--log-dir', required=False, help='Log directory.')
    parser.add_argument('--try-restore', type=argparse_bool, default=False,
                        help='Try to restore existing model. No error if no model found.')
    parser.add_argument('--restore', type=argparse_bool, default=False, help='Whether to restore existing model.')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to restore.')
    parser.add_argument('--config', type=str, help='YAML configuration file.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')

    # Report and checkpoint saving
    parser.add_argument('--io.timeout', type=int, default=10 * 60)
    parser.add_argument('--io.validation_interval', type=int, default=10)
    parser.add_argument('--io.initial_validation_run', type=argparse_bool, default=True)
    parser.add_argument('--io.train_summary_interval', type=int, default=10)
    parser.add_argument('--io.model_summary_interval', type=int, default=10)
    parser.add_argument('--io.test_model_summary_interval', type=int, default=0)
    parser.add_argument('--io.model_summary_num_batches', type=int, default=50)
    parser.add_argument('--io.debug_performance', type=argparse_bool, default=False)
    parser.add_argument('--io.debug_memory', type=argparse_bool, default=False)
    parser.add_argument('--io.debug_numerics', type=argparse_bool, default=False)
    parser.add_argument('--io.debug_queues', type=argparse_bool, default=False)
    parser.add_argument('--io.debug_summary', type=argparse_bool, default=False)
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
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.sample_queue_capacity', type=int, default=1024 * 4)
    parser.add_argument('--tf.sample_queue_min_after_dequeue', type=int, default=2 * 1024)
    parser.add_argument('--tf.use_gpu_feeders', type=argparse_bool, default=True)
    parser.add_argument('--tf.staging_prefetch_group_size', type=int, default=5)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")
    parser.add_argument('--tf.create_tf_timeline', type=argparse_bool, default=False,
                        help="Generate tensorflow trace.")

    # Learning parameters
    parser.add_argument('--training.num_epochs', type=int, default=10000)
    parser.add_argument('--training.batch_size', type=int, default=64)
    parser.add_argument('--training.test_batch_size', type=int, default=-1)
    parser.add_argument('--training.optimizer', type=str, default="adam")
    parser.add_argument('--training.max_grad_global_norm', type=float, default=1e3)
    parser.add_argument('--training.initial_learning_rate', type=float, default=1e-3)
    parser.add_argument('--training.learning_rate_decay_mode', type=str, default="exponential")
    parser.add_argument('--training.learning_rate_decay_epochs', type=int, default=10)
    parser.add_argument('--training.learning_rate_decay_rate', type=float, default=0.96)
    parser.add_argument('--training.learning_rate_decay_staircase', type=argparse_bool, default=False)

    # Data parameters
    parser.add_argument('--data.train_path', required=True, help='Train data path.')
    parser.add_argument('--data.test_path', required=True, help='Test data path.')
    parser.add_argument('--data.type', default="lmdb", help='Type of data storage.')
    parser.add_argument('--data.max_num_batches', type=int, default=-1)
    parser.add_argument('--data.max_num_samples', type=int, default=-1)
    parser.add_argument('--data.use_prefetching', type=argparse_bool, default=True)
    parser.add_argument('--data.prefetch_process_count', type=int, default=2)
    parser.add_argument('--data.prefetch_queue_size', type=int, default=10)
    # Data parameters
    parser.add_argument('--data.fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--data.fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

    args = parser.parse_args()

    run(args)
