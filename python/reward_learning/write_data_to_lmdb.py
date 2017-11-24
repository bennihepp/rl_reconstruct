#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Workaround for segmentation fault for some versions when ndimage is imported after tensorflow.
import scipy.ndimage as nd

import os
import argparse
import file_helpers
import numpy as np
from pybh import pybh_yaml as yaml
from pybh import hdf5_utils, tensorpack_utils, serialization
import data_record
import tensorflow as tf
import tensorpack.dataflow
import tensorpack.utils.serialize
from pybh import lmdb_utils
import input_pipeline
import configuration
import traceback
from pybh.utils import argparse_bool
from pybh.attribute_dict import AttributeDict
from pybh import log_utils


logger = log_utils.get_logger("reward_learning/write_data_to_lmdb")


def write_hdf5_files_to_lmdb(input_and_target_retriever, tf_cfg, hdf5_path, lmdb_path, max_num_files,
                             batch_size, serialization_name, compression, compression_arg, append=False, max_num_samples=None, verbose=False):
    logger.info("Data path: {}".format(hdf5_path))
    filename_generator = file_helpers.input_filename_generator_hdf5(hdf5_path)
    filenames = list(filename_generator)
    # Write filenames to lmdb storage path (for later reference)
    if append:
        file_mode = "a"
    else:
        file_mode = "w"
    with open(lmdb_path + "_filenames", file_mode) as fout:
        for filename in filenames:
            fout.write("{}\n".format(os.path.basename(filename)))

    if len(filenames) == 0:
        raise RuntimeError("No dataset file")
    else:
        logger.info("Found {} dataset files".format(len(filenames)))

    # Limit dataset size?
    if max_num_files > 0:
        filenames = filenames[:max_num_files]
        logger.info("Using {} dataset files".format(len(filenames)))
        # Force recomputation of num of samples if not using all data files
    else:
        logger.info("Using all dataset files")

    if verbose:
        # Retrieve input and target shapes
        tmp_data = input_and_target_retriever.read_data_from_file(filenames[0])
        input_shape = input_and_target_retriever.get_input_from_data(tmp_data).shape[1:]
        target_shape = input_and_target_retriever.get_target_from_data(tmp_data).shape[1:]
        del tmp_data
        logger.info("Input and target shapes:")
        logger.info("  Shape of input: {}".format(input_shape))
        logger.info("  Shape of target: {}".format(target_shape))

    if verbose:
        input_pipeline.print_data_stats(filenames[0], input_and_target_retriever)

    coord = tf.train.Coordinator()

    # Create HDF5 readers
    hdf5_pipeline = hdf5_utils.HDF5ReaderProcessCoordinator(
        filenames, coord, read_data_fn=input_and_target_retriever.read_data_from_file,
        shuffle=False, repeats=1, num_processes=tf_cfg.cpu_queue_processes,
        data_queue_capacity=tf_cfg.cpu_data_queue_capacity, verbose=verbose)

    try:
        custom_threads = []

        hdf5_pipeline.start()
        custom_threads.extend(hdf5_pipeline.threads)

        class HDF5DataFlow(tensorpack.dataflow.DataFlow):

            def __init__(self, hdf5_pipeline, max_num_samples=None):
                self._hdf5_pipeline = hdf5_pipeline
                self._num_samples = 0
                self._max_num_samples = max_num_samples

            def get_data(self):
                input_field_name = input_and_target_retriever.input_field_name
                target_field_name = input_and_target_retriever.target_field_name
                while True:
                    if verbose:
                        logger.debug("Fetching next data block from hdf5 pipeline")
                    full_data = self._hdf5_pipeline.get_next_data()
                    if full_data == hdf5_utils.QUEUE_END:
                        return
                    data = {
                        input_field_name: full_data[input_field_name],
                        target_field_name: full_data[target_field_name],
                    }
                    # TODO: Remove this. Kind of a hack.
                    data = {key: np.asarray(array, dtype=np.float32) for key, array in data.items()}
                    #self._num_samples += data[input_field_name].shape[0]
                    if verbose:
                        logger.debug("Iterating through data block")
                    for i in range(data[input_field_name].shape[0]):
                        sample = [{key: batch[i, ...] for key, batch in data.items()}]
                        yield sample
                        self._num_samples += 1
                        if self._max_num_samples is not None and self._max_num_samples > 0:
                            if self._num_samples >= self._max_num_samples:
                                return

            @property
            def num_samples(self):
                return self._num_samples

        if append:
            logger.info("Appending HDF5 data from {} to lmdb database {}".format(hdf5_path, lmdb_path))
            lmdb_df = tensorpack_utils.AutoLMDBData(lmdb_path, shuffle=False)
            initial_num_samples = lmdb_df.size()
            logger.info("initial_num_samples: {}".format(initial_num_samples))
        else:
            logger.info("Writing HDF5 data from {} to lmdb database {}".format(hdf5_path, lmdb_path))
            initial_num_samples = 0

        if max_num_samples is not None and max_num_samples > 0:
            logger.info("Limiting num of samples to {}".format(max_num_samples))
            max_num_samples = max_num_samples - initial_num_samples
            if max_num_samples < 0:
                logger.info("WARNING: Database already has enough samples")
                return
            if append:
                logger.info("Appending at most {} samples".format(max_num_samples))

        hdf5_df = HDF5DataFlow(hdf5_pipeline, max_num_samples)
        tensorpack_utils.dump_compressed_dataflow_to_lmdb(hdf5_df, lmdb_path, batch_size,
                                                          write_frequency=10,
                                                          serialization_name=serialization_name,
                                                          compression=compression,
                                                          compression_arg=compression_arg,
                                                          append=append)

        if batch_size > 0:
            num_dropped_samples = hdf5_df.num_samples % batch_size
        else:
            num_dropped_samples = 0

        if batch_size > 0:
            lmdb_df = tensorpack_utils.LMDBDataWithMetaData(lmdb_path, shuffle=False)
            logger.info("Database has {} batches".format(lmdb_df.size()))
            logger.info("hdf5_df.num_samples: {}".format(hdf5_df.num_samples))
            logger.info("num_dropped_samples: {}".format(num_dropped_samples))
            logger.info("batch_size: {}".format(batch_size))
            logger.info("lmdb_df.size(): {}".format(lmdb_df.size()))
            if initial_num_samples + hdf5_df.num_samples - num_dropped_samples != batch_size * lmdb_df.size():
                logger.info("initial_num_samples: {}".format(initial_num_samples))
                logger.info("hdf5_df.num_samples: {}".format(hdf5_df.num_samples))
                logger.info("num_dropped_samples: {}".format(num_dropped_samples))
                logger.info("batch_size: {}".format(batch_size))
                logger.info("lmdb_df.size(): {}".format(lmdb_df.size()))
            assert (initial_num_samples + hdf5_df.num_samples - num_dropped_samples == batch_size * lmdb_df.size())

        lmdb_df = tensorpack_utils.AutoLMDBData(lmdb_path, shuffle=False)
        logger.info("Database has {} samples".format(lmdb_df.size()))
        logger.info("hdf5_df.num_samples: {}".format(hdf5_df.num_samples))
        logger.info("num_dropped_samples: {}".format(num_dropped_samples))
        logger.info("batch_size: {}".format(batch_size))
        logger.info("lmdb_df.size(): {}".format(lmdb_df.size()))
        assert (initial_num_samples + hdf5_df.num_samples - num_dropped_samples == lmdb_df.size())

        # Check that we can read data without errors
        lmdb_df.reset_state()
        it = lmdb_df.get_data()
        q = next(it)
        for key in q[0]:
            logger.info(q[0][key].shape)
            logger.info(q[0][key].dtype)
            assert(q[0][key].dtype == np.float32)

        if num_dropped_samples > 0:
            logger.warn("Dropped {} samples from input dataset".format(num_dropped_samples))

        hdf5_pipeline.stop()

    except Exception as exc:
        logger.info("Exception while converting hdf5 data to LMDB database: {}".format(exc))
        traceback.print_exc()
        coord.request_stop(exc)
        raise exc
    finally:
        logger.info("Requesting stop")
        coord.request_stop()
        coord.join(custom_threads, stop_grace_period_secs=10)


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

    cfg = AttributeDict.convert_deep(cfg)

    # Retrieval functions for input and target
    data_stats_dict = {}
    input_and_target_retriever = input_pipeline.InputAndTargetFromHDF5(cfg.data, data_stats_dict, verbose=True)

    if not args.only_compute_stats:
        write_hdf5_files_to_lmdb(input_and_target_retriever, cfg.tensorflow, args.data_path, args.lmdb_output_path,
                                 cfg.data.max_num_files, args.batch_size,
                                 args.serialization, args.compression, args.compression_arg, args.append, args.max_num_samples,
                                 args.verbose)
        logger.info("LMDB database written")

    if args.compute_stats:
        logger.info("Computing data statistics ...")
        lmdb_df = tensorpack_utils.AutoLMDBData(args.lmdb_output_path, shuffle=False)
        lmdb_df.reset_state()

        def dict_from_dataflow_generator(df):
            for sample in df.get_data():
                yield sample[0]

        data_stats_dict = data_record.compute_dataset_stats_from_dicts(dict_from_dataflow_generator(lmdb_df))
        # TODO: Hack to get rid of float64 in HDF5 dataset
        for key in data_stats_dict:
            for key2 in data_stats_dict[key]:
                if data_stats_dict[key][key2] is not None:
                    data_stats_dict[key][key2] = np.asarray(data_stats_dict[key][key2], dtype=np.float32)

        num_samples = data_stats_dict[input_and_target_retriever.input_field_name]["num_samples"]
        assert(num_samples == lmdb_df.size())

        serialization_name = args.serialization
        serializer = serialization.get_serializer_by_name(serialization_name)
        logger.info("Adding data statistics to LMDB databases")
        with lmdb_utils.LMDB(args.lmdb_output_path, readonly=False) as lmdb_db:
            data_stats_dump = serializer.dumps(data_stats_dict)
            lmdb_db.put_item("__stats__", data_stats_dump)


def main():
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Set verbosity level.')
    parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--lmdb-output-path', required=True, help='Path to store LMDB database.')
    parser.add_argument('--config', type=str, help='YAML configuration file.')
    parser.add_argument('--compute-stats', type=argparse_bool, default=True)
    parser.add_argument('--only-compute-stats', type=argparse_bool, default=False)
    parser.add_argument('--serialization', type=str, default="pickle")
    parser.add_argument('--compression', type=str, default="lz4")
    parser.add_argument('--compression-arg', type=str)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-num-samples', type=int)
    parser.add_argument('--append', type=argparse_bool, default=False)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.cpu_data_queue_capacity', type=int, default=10)
    parser.add_argument('--tf.cpu_sample_queue_capacity', type=int, default=1024 * 4)
    parser.add_argument('--tf.cpu_queue_min_after_dequeue', type=int, default=2 * 1024)
    parser.add_argument('--tf.cpu_queue_threads', type=int, default=4)
    parser.add_argument('--tf.cpu_queue_processes', type=int, default=1)

    # Data parameters
    parser.add_argument('--data.max_num_files', type=int, default=-1)
    parser.add_argument('--data.fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--data.fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
