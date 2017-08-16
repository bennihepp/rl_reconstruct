#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
import env_factory
from RLrecon.environments.environment import HorizontalEnvironment
import data_record
import file_helpers


def run():
    filename_prefix = "datasets/16x16x16_1-2-3-4-5"
    filename_template = os.path.join(filename_prefix, file_helpers.DEFAULT_HDF5_TEMPLATE)
    filename_generator = file_helpers.input_filename_generator_hdf5(filename_template)
    i = 0
    while True:
        filename = next(filename_generator, None)
        if filename is None:
            break
        # filename_tf = tf.constant(filename, shape=[1])
        # filename_queue = tf.train.string_input_producer(filename_tf)
        # record = data_record.read_and_decode_tf_example(filename_queue)
        records = data_record.read_hdf5_records_as_list(filename)
        print(filename)
        # for record in records:
        #     print("  ", record.action, record.reward)
        i += 1

if __name__ == '__main__':
    run()
