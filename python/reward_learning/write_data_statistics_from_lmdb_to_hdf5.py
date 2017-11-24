#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Workaround for segmentation fault for some versions when ndimage is imported after tensorflow.
import scipy.ndimage as nd

import argparse
import numpy as np
from pybh import tensorpack_utils
from pybh import hdf5_utils
from pybh import log_utils


logger = log_utils.get_logger("reward_learning/split_data_lmdb")


def dict_from_dataflow_generator(df):
    for sample in df.get_data():
        yield sample[0]


def write_data_statistics_from_lmdb_to_hdf5(lmdb_input_path, hdf5_output_path):
    data_dict_df = tensorpack_utils.AutoLMDBData(lmdb_input_path)
    data_dict_df.reset_state()

    data_stats_dict = data_dict_df.get_metadata("stats")

    hdf5_utils.write_numpy_dict_to_hdf5_file(hdf5_output_path, data_stats_dict)


def run(args):
    write_data_statistics_from_lmdb_to_hdf5(args.lmdb_input_path, args.hdf5_output_path)


def main():
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help='Set verbosity level.')
    parser.add_argument('--lmdb-input-path', required=True, help='Path to input LMDB database.')
    parser.add_argument('--hdf5-output-path', required=True, help='Path to store HDF5 data statistics.')

    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
