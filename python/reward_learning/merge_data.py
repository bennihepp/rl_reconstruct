#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
import data_record
import file_helpers
from utils import argparse_bool


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

    records_per_file = args.records_per_file
    check_written_records = args.check_written_records
    dry_run = args.dry_run

    dataset_kwargs = {}
    if args.compression_level >= 0:
        dataset_kwargs.update({"compression": "gzip",
                               "compression_opts": args.compression_level})

    def write_records(records, next_file_num):
        # filename, next_file_num = get_next_output_tf_filename(next_file_num)
        filename, next_file_num = file_helpers.get_next_output_hdf5_filename(
            next_file_num, template=filename_template)
        print("Writing {} records to file {}".format(len(records), filename))
        if not dry_run:
            data_record.write_hdf5_records_v4(filename, records,
                                              dataset_kwargs=dataset_kwargs)
            if check_written_records:
                print("Reading records from file {}".format(filename))
                records_read = data_record.read_hdf5_records_v4_as_list(filename)
                for record, record_read in zip(records, records_read):
                    assert(np.all(record.intrinsics == record_read.intrinsics))
                    assert(record.map_resolution == record_read.map_resolution)
                    assert(record.axis_mode == record_read.axis_mode)
                    assert(record.forward_factor == record_read.forward_factor)
                    assert(np.all(record.obs_levels == record_read.obs_levels))
                    for in_grid_3d, in_grid_3d_read in zip(record.in_grid_3d, record_read.in_grid_3d):
                        assert(np.all(in_grid_3d == in_grid_3d_read))
                    for out_grid_3d, out_grid_3d_read in zip(record.out_grid_3d, record_read.out_grid_3d):
                        assert(np.all(out_grid_3d == out_grid_3d_read))
                    assert(np.all(record.rewards == record_read.rewards))
                    assert(np.all(record.scores == record_read.scores))
                    assert(np.all(record.rgb_image == record_read.rgb_image))
                    assert(np.all(record.normal_image == record_read.normal_image))
                    assert(np.all(record.depth_image == record_read.depth_image))
        return next_file_num

    total_num_records = 0
    next_file_num = 0
    records = []
    for i, input_file in enumerate(input_files):
        print("Reading input file #{} out of {}".format(i, len(input_files)))
        for record in data_record.generate_single_records_from_hdf5_file_v4(input_file):
            records.append(record)

            if len(records) % records_per_file == 0:
                next_file_num = write_records(records, next_file_num)
                records = []

    if len(records) > 0:
        write_records(records, next_file_num)
        records = []
    assert(len(records) == 0)

    print("Total number of written records: {}".format(total_num_records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--dry-run', action='store_true', help="Do not save anything")
    parser.add_argument('--input-path', type=str, help="Input path")
    parser.add_argument('--input-list-file', type=str, help="File with list of input files")
    parser.add_argument('--output-path', required=True, type=str, help="Output path")
    parser.add_argument('--records-per-file', default=1000, type=int, help="Samples per output file")
    parser.add_argument('--compression-level', default=5, type=int, help="Gzip compression level")
    parser.add_argument('--check-written-records', type=argparse_bool, default=True,
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
