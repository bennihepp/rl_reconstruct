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


def run_episode(episode, environment, input_episode_dict,
                reset_interval=np.iinfo(np.int32).max,
                reset_score_threshold=np.finfo(np.float32).max,
                downsample_to_grid=True,
                measure_timing=False):
    if measure_timing:
        timer = Timer()
        time_meter = TimeMeter()
    else:
        time_meter = DummyTimeMeter()

    pose_list = list([environment.base.Pose(location, orientation_rpy) for location, orientation_rpy \
                      in zip(input_episode_dict["location"], input_episode_dict["orientation_rpy"])])

    intrinsics = environment.base.get_engine().get_intrinsics()

    logger.info("Starting episode")
    environment.reset()

    output = {
        "score": [],
        "computed_reward": [],
        "true_reward": [],
        "pose": [],
    }

    for i in range(len(pose_list)):
        result = environment.base.get_mapper().perform_info()
        score = result.normalized_probabilistic_score
        if score >= reset_score_threshold or i >= reset_interval:
            logger.info("Episode finished with score {} after {} steps".format(score, i))
            print("Episode took {} seconds".format(timer.elapsed_seconds()))
            return output

        current_pose = environment.base.get_pose()

        if i == 0:
            # Initialize output
            output["score"].append(score)
            output["computed_reward"].append(0)
            output["true_reward"].append(0)
            output["pose"].append(current_pose)
            logger.info("Initial pose: {}".format(current_pose))

        logger.info("Step # {} of episode # {}".format(i, episode))

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
        logger.info("Probabilistic reward={}, input reward={}, score={}, input score={}".format(
            reward, input_episode_dict["true_reward"][i], score, input_episode_dict["score"][i]))

        # Record result for this step
        output["score"].append(score)
        output["computed_reward"].append(input_episode_dict["computed_reward"][i])
        output["true_reward"].append(reward)
        output["pose"].append(new_pose)

        # time_meter.print_times()

        i += 1

    print("Episode took {} seconds".format(timer.elapsed_seconds()))

    return output


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

    for i, (input_filename, filename_match) in enumerate(input_filenames_and_matches):
        input_episode_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(os.path.join(input_path, input_filename))
        del pose_list[:]
        pose_list.extend([environment.base.Pose(location, orientation_rpy) for location, orientation_rpy \
                           in zip(input_episode_dict["location"], input_episode_dict["orientation_rpy"])])

        episode = int(filename_match.group(1))

        if args.output_filename_prefix:
            hdf5_filename = "{:s}_{:d}.hdf5".format(
                args.output_filename_prefix,
                episode)
            logger.info("File name for episode {}: {}".format(episode, hdf5_filename))
            if os.path.isfile(hdf5_filename):
                logger.info("File '{}' already exists. Skipping.".format(hdf5_filename))
                continue

        logger.info("Running episode #{} from input file {}".format(episode, input_filename))

        output = run_episode(episode, environment, input_episode_dict,
                             args.reset_interval, args.reset_score_threshold,
                             downsample_to_grid=downsample_to_grid,
                             measure_timing=args.measure_timing)

        if args.output_filename_prefix:
            locations = [pose.location() for pose in output["pose"]]
            orientation_rpys = [pose.orientation_rpy() for pose in output["pose"]]
            hdf5_dict = {
                "score": np.asarray(output["score"]),
                "computed_reward": np.asarray(output["computed_reward"]),
                "true_reward": np.asarray(output["true_reward"]),
                "location": np.asarray(locations),
                "orientation_rpy": np.asarray(orientation_rpys),
            }
            if os.path.isfile(hdf5_filename):
                raise RuntimeError("ERROR: Output file '{}' already exists".format(hdf5_filename))
            if not args.dry_run:
                hdf5_utils.write_numpy_dict_to_hdf5_file(hdf5_filename, hdf5_dict)
                logger.info("Wrote output to HDF5 file: {}".format(hdf5_filename))
        episode += 1


if __name__ == '__main__':
    np.set_printoptions(threshold=50)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--dry-run', type=argparse_bool, default=False)
    parser.add_argument('--measure-timing', type=argparse_bool, default=False)
    parser.add_argument('--input-filename-prefix', type=str, required=True)
    parser.add_argument('--output-filename-prefix', type=str)
    parser.add_argument('--reset-interval', default=100, type=int)
    parser.add_argument('--reset-score-threshold', type=float, default=0.3)

    parser.add_argument('--environment-config', type=str, required=True, help="Environment configuration file")
    parser.add_argument('--client-id', default=0, type=int)

    args = parser.parse_args()

    run(args)
