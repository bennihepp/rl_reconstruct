#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
import data_record
import file_helpers
from pybh.utils import argparse_bool


def run(args):
    if args.input_path is not None:
        input_path = args.input_path
        input_files = list(file_helpers.input_filename_generator_hdf5(input_path, file_helpers.DEFAULT_HDF5_PATTERN))
    else:
        input_list_file = args.input_list_file
        with open(input_list_file, "r") as fin:
            input_files = [l.strip() for l in fin.readlines()]

    print("Merging {} input files".format(len(input_files)))

    output_path1 = args.output_path1
    if not os.path.isdir(output_path1):
        os.makedirs(output_path1)
    filename_template1 = os.path.join(output_path1, file_helpers.DEFAULT_HDF5_TEMPLATE)

    output_path2 = args.output_path2
    if not os.path.isdir(output_path2):
        os.makedirs(output_path2)
    filename_template2 = os.path.join(output_path2, file_helpers.DEFAULT_HDF5_TEMPLATE)

    samples_per_file = args.samples_per_file
    split_ratio1 = args.split_ratio1
    assert(split_ratio1 > 0)
    assert(split_ratio1 < 1)
    dry_run = args.dry_run

    dataset_kwargs = {}
    if args.compression:
        dataset_kwargs.update({"compression": args.compression})
        if args.compression_level >= 0:
            dataset_kwargs.update({"compression_opts": args.compression_level})

    # Count total number of samples
    num_samples = 0
    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        samples = data_record.read_hdf5_records_v4_as_list(input_file)
        num_samples += len(samples)

    num_samples1 = round(split_ratio1 * num_samples)
    num_samples2 = num_samples - num_samples1
    if num_samples1 <= 0 or num_samples2 <= 0:
        import sys
        sys.stderr.write("Data split will result in empty data set\n")
        sys.exit(1)

    def write_samples(samples, next_file_num, filename_template):
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

    # Start with output 1
    filename_template = filename_template1

    finished_output1 = False
    written_samples = 0
    next_file_num = 0
    samples = []
    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        field_dict = None
        samples, attr_dict = data_record.read_samples_from_hdf5_file(input_file, field_dict)
        for sample in samples:
            samples.append(sample)

            do_write_samples = len(samples) % samples_per_file == 0
            if not finished_output1 and written_samples + len(samples) >= num_samples1:
                do_write_samples = True

            if do_write_samples:
                write_samples(samples, next_file_num, filename_template)
                written_samples += len(samples)
                samples = []
            if not finished_output1 and written_samples >= num_samples1:
                finished_output1 = True
                filename_template = filename_template2
                next_file_num = 0

    if len(samples) > 0:
        write_samples(samples, next_file_num, filename_template)
        samples = []
    assert(len(samples) == 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--dry-run', action='store_true', help="Do not save anything")
    parser.add_argument('--input-path', type=str, help="Input path")
    parser.add_argument('--input-list-file', type=str, help="File with list of input files")
    parser.add_argument('--output-path1', required=True, type=str, help="Output path 1")
    parser.add_argument('--output-path2', required=True, type=str, help="Output path 2")
    parser.add_argument('--samples-per-file', default=1000, type=int, help="Samples per output file")
    parser.add_argument('--split-ratio1', default=0.8, type=float, help="Ratio of data to write to output path 1")
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
