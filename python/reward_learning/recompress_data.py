#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import argparse
import file_helpers
import numpy as np
from pybh.utils import argparse_bool
from pybh import hdf5_utils


def run(args):
    if args.input_path is not None:
        input_path = args.input_path
        input_files = list(file_helpers.input_filename_generator_hdf5(input_path, file_helpers.DEFAULT_HDF5_PATTERN))
    else:
        input_list_file = args.input_list_file
        with file(input_list_file, "r") as fin:
            input_files = [l.strip() for l in fin.readlines()]

    dataset_kwargs = {}
    if args.compression:
        dataset_kwargs.update({"compression": args.compression})
        if args.compression_level >= 0:
            dataset_kwargs.update({"compression_opts": args.compression_level})

    print("Counting {} input files".format(len(input_files)))

    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        field_dict = None
        data, attr_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(input_file, field_dict, read_attributes=True)
        print("Writing {} samples with new compression settings".format(data[list(data.keys())[0]].shape[0]))
        hdf5_utils.write_numpy_dict_to_hdf5_file(input_file + "_recompressed", data, attr_dict, **dataset_kwargs)
        if args.check_written_samples:
            print("Reading samples from file {}".format(input_file + "_recompressed"))
            written_data, written_attr_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(input_file + "_recompressed",
                                                                                      field_dict, read_attributes=True)
            for key in data:
                assert(np.all(data[key] == written_data[key]))
            for key in attr_dict:
                assert(np.all(attr_dict[key] == written_attr_dict[key]))
        os.remove(input_file)
        os.rename(input_file + "_recompressed", input_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--input-path', type=str, help="Input path")
    parser.add_argument('--input-list-file', type=str, help="File with list of input files")
    parser.add_argument('--compression', type=str, default="gzip", help="Type of compression to use. Default is gzip.")
    parser.add_argument('--compression-level', default=5, type=int, help="Gzip compression level")
    parser.add_argument('--check-written-samples', type=argparse_bool, default=True,
                        help="Whether written files should be read and checked afterwards.")

    args = parser.parse_args()

    if args.input_path is not None:
        pass
    elif args.input_list_file is not None:
        pass
    else:
        import sys
        sys.stderr.write("ERROR: One of --input-path or input-list-file options has to be provided.\n")
        parser.print_help()
        sys.exit(1)

    run(args)
