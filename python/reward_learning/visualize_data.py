#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import numpy as np
import mayavi.mlab as mlab
import data_record
import file_helpers


def get_xyz_grids(grid, scale=1.0):
    x, y, z = scale * np.mgrid[:grid.shape[0], :grid.shape[1], :grid.shape[2]]
    return x, y, z


def create_mlab_figure(width=640, height=480):
    return mlab.figure(size=(width, height))


def get_point_cloud_extent(x, y, z, margin=0.1):
    x_extent = np.max(x) - np.min(x)
    y_extent = np.max(y) - np.min(y)
    z_extent = np.max(z) - np.min(z)
    x_center = (np.min(x) + np.max(x)) / 2.
    y_center = (np.min(y) + np.max(y)) / 2.
    z_center = (np.min(z) + np.max(z)) / 2.
    half_margin_factor = 0.5 + 0.5 * margin
    extent = [x_center - half_margin_factor * x_extent, x_center + half_margin_factor * x_extent,
              y_center - half_margin_factor * y_extent, y_center + half_margin_factor * y_extent,
              z_center - half_margin_factor * z_extent, z_center + half_margin_factor * z_extent]
    return extent


def mask_point_cloud(mask, *args):
    flat_mask = mask.flatten()
    masked_args = []
    for arg in args:
        masked_arg = arg.flatten()[flat_mask]
        masked_args.append(masked_arg)
    return masked_args


def plot3d_with_mask(x, y, z, s, mask, opacity=1.0, colormap='hot', mode='cube',
                     vmin=0., vmax=1., fig=None):
    mlab.points3d(x, y, z, s,
                  colormap=colormap,
                  figure=fig,
                  opacity=0.1 * opacity,
                  scale_mode='none',
                  mode=mode,
                  scale_factor=0.02,
                  vmin=vmin,
                  vmax=vmax)
    x, y, z, s = mask_point_cloud(mask, x, y, z, s)
    mlab.points3d(x, y, z, s,
                  colormap=colormap,
                  figure=fig,
                  opacity=opacity,
                  scale_mode='none',
                  mode=mode,
                  scale_factor=0.05,
                  vmin=vmin,
                  vmax=vmax)


def plot3d_with_threshold(x, y, z, s, low_thres=0.0, high_thres=1.0, opacity=1.0, colormap='hot', mode='cube',
                          vmin=0., vmax=1., fig=None):
    mask = np.logical_and(s >= low_thres, s <= high_thres)
    plot3d_with_mask(x, y, z, s, mask, opacity=opacity, colormap=colormap, mode=mode, vmin=vmin, vmax=vmax, fig=fig)


def contour3d(x, y, z, s, contours=[0.5], opacity=1.0, colormap='hot',
               vmin=0., vmax=1., fig=None):
    if contours is not None and len(contours) > 0:
        mlab.contour3d(x, y, z, s,
                       contours=contours,
                       colormap=colormap,
                       figure=fig,
                       opacity=opacity,
                       vmin=vmin,
                       vmax=vmax)
    else:
        mlab.contour3d(x, y, z, s,
                       colormap=colormap,
                       figure=fig,
                       opacity=opacity,
                       vmin=vmin,
                       vmax=vmax)


def plot_grid(occupancies_3d, observation_counts_3d,
              title_prefix="", show=True):
    x, y, z = get_xyz_grids(occupancies_3d, scale=0.1)

    s = occupancies_3d
    fig = create_mlab_figure()
    plot3d_with_threshold(x, y, z, s, low_thres=0.6, fig=fig)
    mlab.title(title_prefix + "Occupied")
    mlab.scalarbar()

    # s = occupancies_3d
    # fig = create_mlab_figure()
    # plot3d_with_threshold(x, y, z, s, fig=fig)
    # mlab.title("Occupancy")
    # mlab.scalarbar()

    # fig = create_mlab_figure()
    contour3d(x, y, z, occupancies_3d, contours=[0.6], opacity=0.7, fig=fig)

    s = occupancies_3d
    fig = create_mlab_figure()
    mask = np.logical_and(s <= 0.4, observation_counts_3d < 1.0)
    plot3d_with_mask(x, y, z, s, mask, fig=fig)
    mlab.title(title_prefix + "Free")
    mlab.scalarbar()

    s = observation_counts_3d
    fig = create_mlab_figure()
    plot3d_with_threshold(x, y, z, s, high_thres=0.9, fig=fig)
    mlab.title(title_prefix + "Observation count")
    mlab.scalarbar()

    # s = observation_counts_3d
    # fig = create_mlab_figure()
    # plot3d_with_threshold(x, y, z, s, fig=fig)
    # mlab.title("Observation count2")
    # mlab.scalarbar()

    if show:
        mlab.show()


def plot_diff_grid(occupancies_3d, observation_counts_3d,
                   epsilon=1e-5, title_prefix="", show=True):
    x, y, z = get_xyz_grids(occupancies_3d, scale=0.1)

    s = occupancies_3d
    fig = create_mlab_figure()
    mask = np.abs(s) > epsilon
    print(np.max(np.abs(s)))
    print(np.min(np.abs(s[mask])))
    plot3d_with_mask(x, y, z, s, mask, fig=fig)
    mlab.title(title_prefix + "Occupancy")
    mlab.scalarbar()

    s = observation_counts_3d
    fig = create_mlab_figure()
    mask = np.abs(s) > epsilon
    print(np.max(np.abs(s)))
    print(np.min(np.abs(s[mask])))
    plot3d_with_mask(x, y, z, s, mask, fig=fig)
    mlab.title(title_prefix + "Observation count")
    mlab.scalarbar()

    if show:
        mlab.show()


def run(args):
    filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
    filename = next(filename_generator)
    records = data_record.read_hdf5_records_v3_as_list(filename)
    record_index = np.random.choice(len(records))
    record = records[record_index]
    in_grid_3d = record.in_grid_3d[..., 2:4]
    plot_grid(in_grid_3d[..., 0], in_grid_3d[..., 1], title_prefix="In ", show=False)
    out_grid_3d = record.out_grid_3d[..., 2:4]
    plot_grid(out_grid_3d[..., 0], out_grid_3d[..., 1], title_prefix="Out ", show=False)

    diff_grid_3d = out_grid_3d - in_grid_3d
    plot_diff_grid(diff_grid_3d[..., 0], diff_grid_3d[..., 1], title_prefix="Diff ", show=False)

    mlab.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    # parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--data-path', default="/mnt/data1/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_SimpleV2Environment_greedy", help='Data path.')

    args = parser.parse_args()

    run(args)
