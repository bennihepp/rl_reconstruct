#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Workaround for segmentation fault for some versions when ndimage is imported after tensorflow.
import scipy.ndimage as nd

import argparse
import numpy as np
from pybh import tensorpack_utils
import data_record
from pybh import serialization
from pybh import msgpack_utils
from pybh import lmdb_utils
from pybh.utils import argparse_bool, logged_time_measurement
from pybh import log_utils


logger = log_utils.get_logger("reward_learning/split_data_lmdb")


def dict_from_dataflow_generator(df):
    for sample in df.get_data():
        yield sample[0]


def split_lmdb_dataset(lmdb_input_path, lmdb_output_path1, lmdb_output_path2, split_ratio1,
                       batch_size, shuffle, serialization_name, compression, compression_arg, max_num_samples=None):
    data_dict_df = tensorpack_utils.AutoLMDBData(lmdb_input_path, shuffle=shuffle)
    data_dict_df.reset_state()

    assert(split_ratio1 > 0)
    assert(split_ratio1 < 1)

    num_samples = data_dict_df.size()
    if max_num_samples is not None and max_num_samples > 0:
        num_samples = min(num_samples, max_num_samples)
    num_batches = num_samples // batch_size
    num_batches1 = round(split_ratio1 * num_samples) // batch_size
    num_samples1 = num_batches1 * batch_size
    num_batches2 = num_batches - num_batches1
    num_samples2 = num_batches2 * batch_size
    if num_samples1 <= 0 or num_samples2 <= 0:
        import sys
        sys.stderr.write("Data split will result in empty data set\n")
        sys.exit(1)

    logger.info("Splitting {} samples into {} train and {} test samples".format(num_samples, num_samples1, num_samples2))
    if num_samples > num_samples1 + num_samples2:
        logger.warn("Dropping {} samples from input dataset".format(num_samples - num_samples1 - num_samples2))

    fixed_size_df = tensorpack_utils.FixedSizeData(data_dict_df, size=num_samples1, keep_state=True)
    with logged_time_measurement(logger, "Writing train dataset to {} ...".format(lmdb_output_path1), log_start=True):
        tensorpack_utils.dump_compressed_dataflow_to_lmdb(fixed_size_df, lmdb_output_path1, batch_size,
                                                          write_frequency=10,
                                                          serialization_name=serialization_name,
                                                          compression=compression,
                                                          compression_arg=compression_arg)

    fixed_size_df.set_size(num_samples2)
    with logged_time_measurement(logger, "Writing test dataset to {} ...".format(lmdb_output_path2), log_start=True):
        tensorpack_utils.dump_compressed_dataflow_to_lmdb(fixed_size_df, lmdb_output_path2, batch_size,
                                                          write_frequency=10,
                                                          serialization_name=serialization_name,
                                                          compression=compression,
                                                          compression_arg=compression_arg,
                                                          reset_df_state=False)

    logger.info("Tagging as train and test")
    with lmdb_utils.LMDB(lmdb_output_path1, readonly=False) as lmdb_db:
        lmdb_db.put_item("__train__", msgpack_utils.dumps(True))
    with lmdb_utils.LMDB(lmdb_output_path2, readonly=False) as lmdb_db:
        lmdb_db.put_item("__test__", msgpack_utils.dumps(True))

    lmdb_df = tensorpack_utils.AutoLMDBData(lmdb_output_path1)
    assert(lmdb_df.size() == num_samples1)
    lmdb_df = tensorpack_utils.AutoLMDBData(lmdb_output_path2)
    assert(lmdb_df.size() == num_samples2)


def compute_and_update_stats_in_lmdb(lmdb_path, serialization_name):
    with logged_time_measurement(logger, "Computing data statistics for {}".format(lmdb_path), log_start=True):
        lmdb_df = tensorpack_utils.AutoLMDBData(lmdb_path)
        lmdb_df.reset_state()
        data_stats_dict = data_record.compute_dataset_stats_from_dicts(dict_from_dataflow_generator(lmdb_df))

    # TODO: Hack to get rid of float64 in HDF5 dataset
    for key in data_stats_dict:
        for key2 in data_stats_dict[key]:
            if data_stats_dict[key][key2] is not None:
                data_stats_dict[key][key2] = np.asarray(data_stats_dict[key][key2], dtype=np.float32)

    serializer = serialization.get_serializer_by_name(serialization_name)
    logger.info("Writing data statistics to {}".format(lmdb_path))
    with lmdb_utils.LMDB(lmdb_path, readonly=False) as lmdb_db:
        data_stats_dump = serializer.dumps(data_stats_dict)
        lmdb_db.put_item("__stats__", data_stats_dump)


def run(args):
    split_lmdb_dataset(args.lmdb_input_path, args.lmdb_output_path1, args.lmdb_output_path2,
                       args.split_ratio1, args.batch_size,
                       args.shuffle, args.serialization,
                       args.compression, args.compression_arg,
                       args.max_num_samples)

    if args.compute_stats:
        compute_and_update_stats_in_lmdb(args.lmdb_output_path1, args.serialization)
        compute_and_update_stats_in_lmdb(args.lmdb_output_path2, args.serialization)


def main():
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Set verbosity level.')
    parser.add_argument('--lmdb-input-path', required=True, help='Path to input LMDB database.')
    parser.add_argument('--lmdb-output-path1', required=True, help='Path to store train LMDB database.')
    parser.add_argument('--lmdb-output-path2', required=True, help='Path to store test LMDB database.')
    parser.add_argument('--shuffle', type=argparse_bool, default=True)
    parser.add_argument('--serialization', type=str, default="pickle")
    parser.add_argument('--compression', type=str, default="lz4")
    parser.add_argument('--compression-arg', type=str)
    parser.add_argument('--split-ratio1', default=0.8, type=float, help="Ratio of data to write to output path 1")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--compute-stats', type=argparse_bool, default=True)
    parser.add_argument('--max-num-samples', type=int)

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
