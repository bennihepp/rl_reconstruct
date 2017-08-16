#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np
import mayavi.mlab as mlab
import data_record
import file_helpers
from visualization import plot_grid, plot_diff_grid


def run(args):
    filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
    filenames = list(filename_generator)
    np.random.shuffle(filenames)
    for filename in filenames:
        print("Filename {}".format(filename))
        records = data_record.read_hdf5_records_v3_as_list(filename)
        indices = np.arange(len(records))
        np.random.shuffle(indices)
        for index in indices:
            print("  record # {}".format(index))
            record = records[index]
            in_grid_3d = record.in_grid_3d[..., 2:4]
            fig = 1
            fig = plot_grid(in_grid_3d[..., 0], in_grid_3d[..., 1], title_prefix="In ", show=False, fig_offset=fig)
            out_grid_3d = record.out_grid_3d[..., 2:4]
            fig = plot_grid(out_grid_3d[..., 0], out_grid_3d[..., 1], title_prefix="Out ", show=False, fig_offset=fig)

            diff_grid_3d = out_grid_3d - in_grid_3d
            fig = plot_diff_grid(diff_grid_3d[..., 0], diff_grid_3d[..., 1], title_prefix="Diff ", show=False, fig_offset=fig)

            mlab.show(stop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    # parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--data-path', default="/mnt/data1/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_SimpleV2Environment_greedy", help='Data path.')

    args = parser.parse_args()

    run(args)
