#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
import data_record
from pybh import hdf5_utils
import file_helpers
from pybh.utils import argparse_bool


def run(args):
    if args.input_path is not None:
        input_path = args.input_path
        input_files = list(file_helpers.input_filename_generator_hdf5(input_path, file_helpers.DEFAULT_HDF5_PATTERN))
    else:
        input_list_file = args.input_list_file
        with file(input_list_file, "r") as fin:
            input_files = [l.strip() for l in fin.readlines()]

    print("Merging {} input files".format(len(input_files)))

    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    filename_template = os.path.join(output_path, file_helpers.DEFAULT_HDF5_TEMPLATE)

    samples_per_file = args.samples_per_file
    check_written_samples = args.check_written_samples
    dry_run = args.dry_run

    dataset_kwargs = {}
    if args.compression:
        dataset_kwargs.update({"compression": args.compression})
        if args.compression_level >= 0:
            dataset_kwargs.update({"compression_opts": args.compression_level})

    def write_samples(samples, next_file_num):
        # filename, next_file_num = get_next_output_tf_filename(next_file_num)
        filename, next_file_num = file_helpers.get_next_output_hdf5_filename(
            next_file_num, template=filename_template)
        print("Writing {} samples to file {}".format(len(samples), filename))
        if not dry_run:
            data_record.write_samples_to_hdf5_file(filename, samples, attr_dict, **dataset_kwargs)
            if args.check_written_samples:
                print("Reading samples from file {}".format(filename))
                written_samples, written_attr_dict = data_record.read_samples_from_hdf5_file(filename,
                                                                                             read_attributes=True)
                assert(len(samples) == len(written_samples))
                for i in range(len(samples)):
                    for key in sample[i]:
                        assert(np.all(samples[i][key] == written_samples[i][key]))
                for key in attr_dict:
                    assert(np.all(attr_dict[key] == written_attr_dict[key]))
        return next_file_num

    total_num_samples = 0
    next_file_num = 0
    samples = []
    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        field_dict = None
        samples, attr_dict = data_record.read_samples_from_hdf5_file(input_file, field_dict)
        for sample in samples:
            samples.append(sample)

            if len(samples) % samples_per_file == 0:
                next_file_num = write_samples(samples, next_file_num)
                samples = []

    if len(samples) > 0:
        write_samples(samples, next_file_num)
        samples = []
    assert(len(samples) == 0)

    print("Total number of written samples: {}".format(total_num_samples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--dry-run', action='store_true', help="Do not save anything")
    parser.add_argument('--input-path', type=str, help="Input path")
    parser.add_argument('--input-list-file', type=str, help="File with list of input files")
    parser.add_argument('--output-path', required=True, type=str, help="Output path")
    parser.add_argument('--samples-per-file', default=1000, type=int, help="Samples per output file")
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
