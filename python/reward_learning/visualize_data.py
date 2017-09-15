#!/usr/bin/env python
from __future__ import print_function

import argparse
import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
import cv2
import data_record
import file_helpers
from visualization import plot_grid, plot_diff_grid


def run(args):
    filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
    filenames = list(filename_generator)
    np.random.shuffle(filenames)
    plt.ion()
    for filename in filenames:
        print("Filename {}".format(filename))
        records = data_record.read_hdf5_records_v4_as_list(filename)
        indices = np.arange(len(records))
        np.random.shuffle(indices)
        for index in indices:
            print("  record # {}".format(index))
            record = records[index]

            depth_image = record.depth_image
            print(np.mean(depth_image))
            if depth_image.shape[0] == 1:
                fig = plt.figure(1)
                plt.clf()
                # plt.plot(np.arange(depth_image.shape[1]), depth_image[0, ...])
                plt.step(np.arange(depth_image.shape[1]), depth_image[0, ...])
                plt.title("Depth image")
                fig.canvas.draw()
                plt.show(block=False)
            else:
                cv2.imshow("depth_image", depth_image / np.max(depth_image))
                cv2.waitKey(50)

            in_grid_3d = record.in_grid_3d[..., 2:4]
            fig = 1
            fig = plot_grid(in_grid_3d[..., 0], in_grid_3d[..., 1], title_prefix="In ", show=False, fig_offset=fig)
            out_grid_3d = record.out_grid_3d[..., 2:4]
            fig = plot_grid(out_grid_3d[..., 0], out_grid_3d[..., 1], title_prefix="Out ", show=False, fig_offset=fig)

            diff_grid_3d = out_grid_3d - in_grid_3d
            plot_diff_grid(diff_grid_3d[..., 0], diff_grid_3d[..., 1], title_prefix="Diff ", show=False, fig_offset=fig)
            print("squared average diff occupancy: {}".format(np.mean(np.square(diff_grid_3d[..., 0]))))
            print("squared average diff observation count: {}".format(np.mean(np.square(diff_grid_3d[..., 1]))))

            mlab.show(stop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    # parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--data-path', default="/home/bhepp/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/data_0", help='Data path.')

    args = parser.parse_args()

    run(args)
