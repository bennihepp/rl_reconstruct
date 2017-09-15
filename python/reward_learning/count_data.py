#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import argparse
import data_record
import file_helpers


def run(args):
    if args.input_path is not None:
        input_path = args.input_path
        input_files = list(file_helpers.input_filename_generator_hdf5(input_path, file_helpers.DEFAULT_HDF5_PATTERN))
    else:
        input_list_file = args.input_list_file
        with file(input_list_file, "r") as fin:
            input_files = [l.strip() for l in fin.readlines()]

    print("Counting {} input files".format(len(input_files)))

    # Count total number of records
    total_num_records = 0
    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        records = data_record.read_hdf5_records_v4_as_list(input_file)
        total_num_records += len(records)

    print("Total number of records: {}".format(total_num_records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--input-path', type=str, help="Input path")
    parser.add_argument('--input-list-file', type=str, help="File with list of input files")

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
