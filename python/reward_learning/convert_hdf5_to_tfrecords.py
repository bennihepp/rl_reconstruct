#!/usr/bin/env python

#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import data_record
import file_helpers
import numpy as np
import tensorflow as tf
import data_record


def run(args):
    filename_generator = file_helpers.input_filename_generator_hdf5(args.hdf5_path)
    filenames = list(filename_generator)

    if not os.path.isdir(args.tfrecords_path):
        os.makedirs(args.tfrecords_path)
    for filename in filenames:
        hdf5_record_batch = data_record.read_hdf5_records_v2(filename)
        output_filename = os.path.join(args.tfrecords_path, os.path.basename(filename[:-len('.hdf5')]) + '.tfrecords')
        print(output_filename)
        tfrecords_writer = tf.python_io.TFRecordWriter(output_filename)
        for example in data_record.generate_tfrecords_from_batch(hdf5_record_batch):
            tfrecords_writer.write(example.SerializeToString())
        tfrecords_writer.close()


if __name__ == '__main__':
    np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--hdf5-path', required=True, help='HDF5 input path.')
    parser.add_argument('--tfrecords-path', required=True, help='TFrecords output path.')

    args = parser.parse_args()

    run(args)
