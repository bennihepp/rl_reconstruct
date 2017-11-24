#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
from builtins import range
import numpy as np
import yaml
import env_factory
from pybh.utils import Timer, TimeMeter, DummyTimeMeter, argparse_bool
from pybh import hdf5_utils
from pybh import file_utils
from pybh import log_utils


logger = log_utils.get_logger("evaluate_reward_trajectory")


def run_episode(environment, pose_list,
                downsample_to_grid=True,
                measure_timing=False):
    timer = Timer()
    if measure_timing:
        time_meter = TimeMeter()
    else:
        time_meter = DummyTimeMeter()

    intrinsics = environment.base.get_engine().get_intrinsics()

    logger.info("Starting episode")
    environment.reset()

    current_pose = environment.base.get_pose()
    logger.info("Initial pose: {}".format(current_pose))

    for i in range(len(pose_list)):
        # result = environment.base.get_mapper().perform_info()
        # score = result.normalized_probabilistic_score

        logger.info("Step # {}".format(i))

        new_pose = pose_list[i]
        with time_meter.measure("set_pose"):
            environment.base.set_pose(new_pose, wait_until_set=True)
        with time_meter.measure("get_depth_image"):
            # environment.base.get_engine().get_depth_image()
            depth_image = environment.base.get_engine().get_depth_image()

        with time_meter.measure("insert_depth_image"):
            result = environment.base.get_mapper().perform_insert_depth_map_rpy(
                new_pose.location(), new_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)

        # Output expected and received reward
        score = result.normalized_probabilistic_score
        reward = result.probabilistic_reward
        logger.info("Probabilistic reward={}, score={}".format(reward, score))

    print("Episode took {} seconds".format(timer.elapsed_seconds()))


def run(args):
    # Create environment
    client_id = args.client_id
    with open(args.environment_config, "r") as fin:
        environment_config = yaml.load(fin)
    environment = env_factory.create_environment_from_config(environment_config, client_id,
                                                             use_openai_wrapper=True)

    result = environment.base.get_mapper().perform_info()
    map_resolution = result.resolution
    axis_mode = environment_config["collect_data"]["axis_mode"]
    forward_factor = float(environment_config["collect_data"]["forward_factor"])
    downsample_to_grid = environment_config["collect_data"]["downsample_to_grid"]
    raycast_max_range = float(environment_config["octomap"]["max_range"])
    logger.info("map_resolution={}".format(map_resolution))
    logger.info("axis_mode={}".format(axis_mode))
    logger.info("forward_factor={}".format(forward_factor))
    logger.info("downsample_to_grid={}".format(downsample_to_grid))
    logger.info("raycast_max_range={}".format(raycast_max_range))

    environment.base.get_engine().disable_input()

    pose_list = []

    def before_reset_hook(env):
        pose = pose_list[0]
        print("Resetting episode with pose {}".format(pose))
        return pose

    def after_reset_hook(env):
        logger.info("Env reset in pose {}".format(env.base.get_pose()))

    environment.base.before_reset_hooks.register(before_reset_hook, environment)
    environment.base.after_reset_hooks.register(after_reset_hook, environment)

    input_path = os.path.dirname(args.input_filename_prefix)
    print("Input path: {}".format(input_path))
    input_filename_pattern = "{:s}_(\d+)\.hdf5".format(os.path.basename(args.input_filename_prefix))
    input_filenames_and_matches = file_utils.get_matching_filenames(input_filename_pattern,
                                                                    path=input_path, return_match_objects=True)
    print("Number of input files: {}".format(len(input_filenames_and_matches)))

    only_episode = args.episode
    max_steps = args.max_steps

    for i, (input_filename, filename_match) in enumerate(input_filenames_and_matches):
        input_episode_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(os.path.join(input_path, input_filename))
        episode_poses = [environment.base.Pose(location, orientation_rpy) for location, orientation_rpy \
                          in zip(input_episode_dict["location"], input_episode_dict["orientation_rpy"])]
        if max_steps is not None:
            episode_poses = episode_poses[:max_steps]

        del pose_list[:]
        pose_list.extend(episode_poses)

        episode = int(filename_match.group(1))
        if only_episode is not None and episode != only_episode:
            continue

        logger.info("Running episode #{} from input file {}".format(episode, input_filename))

        run_episode(environment, pose_list,
                    downsample_to_grid=downsample_to_grid,
                    measure_timing=args.measure_timing)

        episode += 1


if __name__ == '__main__':
    np.set_printoptions(threshold=50)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--measure-timing', type=argparse_bool, default=False)
    parser.add_argument('--input-filename-prefix', type=str, required=True)

    parser.add_argument('--environment-config', type=str, required=True, help="Environment configuration file")
    parser.add_argument('--client-id', default=0, type=int)

    parser.add_argument('--episode', type=int)
    parser.add_argument('--max-steps', type=int)

    args = parser.parse_args()

    run(args)
