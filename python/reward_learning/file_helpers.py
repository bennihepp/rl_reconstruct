import os
import re

DEFAULT_TFRECORDS_TEMPLATE = "data_{:04d}.tfrecords"
DEFAULT_HDF5_TEMPLATE = "data_{:04d}.hdf5"

DEFAULT_TFRECORDS_PATTERN = "data_\d+.tfrecords"
DEFAULT_HDF5_PATTERN = "data_\d+.hdf5"
DEFAULT_HDF5_STATS_FILENAME = "data_stats.hdf5"


def get_tf_filename(file_num, template=DEFAULT_TFRECORDS_TEMPLATE):
    return template.format(file_num)


def get_hdf5_filename(file_num, template=DEFAULT_HDF5_TEMPLATE):
    return template.format(file_num)


# def input_filename_generator(template, start_file_num=0):
#     file_num = start_file_num
#     while True:
#         filename = template.format(file_num)
#         if not os.path.isfile(filename):
#             break
#         yield filename
#         file_num += 1
#
#
# def input_filename_generator_tf(template=DEFAULT_TFRECORDS_TEMPLATE, start_file_num=0):
#     return input_filename_generator(template, start_file_num)
#
#
# def input_filename_generator_hdf5(template=DEFAULT_HDF5_TEMPLATE, start_file_num=0):
#     return input_filename_generator(template, start_file_num)


def input_filename_generator(path, re_pattern=None):
    filenames = os.listdir(path)
    if re_pattern is not None:
        re_pattern = re.compile(re_pattern)
    for filename in filenames:
        if re_pattern is None or re_pattern.match(filename):
            yield os.path.join(path, filename)


def input_filename_generator_tfrecords(path, re_pattern=DEFAULT_TFRECORDS_PATTERN):
    return input_filename_generator(path, re_pattern)


def input_filename_generator_hdf5(path, re_pattern=DEFAULT_HDF5_PATTERN):
    return input_filename_generator(path, re_pattern)


def get_next_output_filename(next_file_num, filename_template):
    while True:
        filename = filename_template.format(next_file_num)
        if not os.path.isfile(filename):
            break
        next_file_num += 1
    return filename, next_file_num


def get_next_output_tf_filename(next_file_num, template=DEFAULT_TFRECORDS_TEMPLATE):
    return get_next_output_filename(next_file_num, template)


def get_next_output_hdf5_filename(next_file_num, template=DEFAULT_HDF5_TEMPLATE):
    return get_next_output_filename(next_file_num, template)
