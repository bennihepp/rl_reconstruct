#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import numpy as np
import data_record
import env_factory
import file_helpers
import RLrecon.environments.environment as RLenvironment

import matplotlib.pyplot as plt


def get_environment_class_by_name(environment_name):
    if environment_name == "HorizontalEnvironment":
        environment_class = RLenvironment.HorizontalEnvironment
    elif environment_name == "SimpleV0Environment":
        environment_class = RLenvironment.SimpleV0Environment
    elif environment_name == "SimpleV2Environment":
        environment_class = RLenvironment.SimpleV2Environment
    else:
        raise NotImplementedError("Unknown environment class: {}".format(environment_name))
    return environment_class


def run(args):
    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    filename_template = os.path.join(output_path, file_helpers.DEFAULT_HDF5_TEMPLATE)

    num_records = args.num_records
    records_per_file = 1000
    reset_interval = 100
    reset_score_threshold = 0.5
    check_written_records = False

    epsilon = 0.1

    obs_levels = [0, 1, 2, 3]
    obs_sizes_x = [16] * len(obs_levels)
    obs_sizes_y = obs_sizes_x
    obs_sizes_z = obs_sizes_x

    environment_class = get_environment_class_by_name(args.environment)
    client_id = args.client_id
    environment = env_factory.create_environment(environment_class, client_id)

    # environment.get_engine().test()

    intrinsics = environment.get_engine().get_intrinsics()

    current_file_num = 0
    records = []
    environment.reset()
    prev_action = np.random.randint(0, environment.get_num_of_actions())
    for i in xrange(num_records):
        print("Record #{}".format(i))
        current_pose = environment.get_pose()

        result = environment.get_mapper().perform_info()
        score = result.score
        normalized_score = result.normalized_score
        prob_score = result.probabilistic_score
        normalized_prob_score = result.normalized_probabilistic_score
        scores = np.array([score, normalized_score, prob_score, normalized_prob_score])

        print("  scores: {}".format(scores))

        if i % reset_interval == 0 or prob_score >= reset_score_threshold:
            environment.reset()

        new_poses = []
        depth_images = []

        # Query octomap
        grid_3ds = None
        for k in xrange(len(obs_levels)):
            obs_level = obs_levels[k]
            obs_size_x = obs_sizes_x[k]
            obs_size_y = obs_sizes_y[k]
            obs_size_z = obs_sizes_z[k]
            res = environment.get_mapper().perform_query_subvolume_rpy(
                current_pose.location(), current_pose.orientation_rpy(),
                obs_level, obs_size_x, obs_size_y, obs_size_z)
            occupancies = np.asarray(res.occupancies, dtype=np.float32)
            occupancies_3d = np.reshape(occupancies, (obs_size_x, obs_size_y, obs_size_z))
            observation_counts = np.asarray(res.observation_counts, dtype=np.float32)
            # observation_counts /= (10.0 * 2 ** obs_level)
            # observation_counts = np.minimum(observation_counts, 10.0)
            observation_counts_3d = np.reshape(observation_counts, (obs_size_x, obs_size_y, obs_size_z))
            # Plot histograms
            # num_bins = 50
            # plt.figure()
            # plt.hist(occupancies_3d.flatten(), num_bins)
            # plt.title("occupancies level {}".format(obs_level))
            # plt.figure()
            # plt.hist(observation_counts.flatten(), num_bins)
            # plt.title("observation counts level {}".format(obs_level))
            # print("Stats for occupancies level {}".format(obs_level))
            # print("  Mean: {}\n  Stddev: {}\n  Min: {}\n  Max: {}".format(
            #     np.mean(occupancies_3d.flatten()),
            #     np.std(occupancies_3d.flatten()),
            #     np.min(occupancies_3d.flatten()),
            #     np.max(occupancies_3d.flatten()),
            # ))
            # print("Stats for observation count level {}".format(obs_level))
            # print("  Mean: {}\n  Stddev: {}\n  Min: {}\n  Max: {}".format(
            #     np.mean(observation_counts.flatten()),
            #     np.std(observation_counts.flatten()),
            #     np.min(observation_counts.flatten()),
            #     np.max(observation_counts.flatten()),
            # ))
            grid_3d = np.stack([occupancies_3d, observation_counts_3d], axis=-1)
            if grid_3ds is None:
                grid_3ds = grid_3d
            else:
                grid_3ds = np.concatenate([grid_3ds, grid_3d], axis=-1)
        # plt.show()

        # Simulate effect of actions and compute depth maps and rewards
        results = []
        rewards = np.zeros((environment.get_num_of_actions(),))
        norm_rewards = np.zeros((environment.get_num_of_actions(),))
        prob_rewards = np.zeros((environment.get_num_of_actions(),))
        norm_prob_rewards = np.zeros((environment.get_num_of_actions(),))
        for action in xrange(environment.get_num_of_actions()):
            new_pose = environment.simulate_action_on_pose(current_pose, action)
            environment.set_pose(new_pose, wait_until_set=True)
            # point_cloud = environment._get_depth_point_cloud(new_pose)
            # result = environment.get_mapper().perform_insert_point_cloud_rpy(
            #     new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
            depth_image = environment.get_engine().get_depth_image()
            new_poses.append(new_pose)
            depth_images.append(depth_image)
            result = environment.get_mapper().perform_insert_depth_map_rpy(
                new_pose.location(), new_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=True, simulate=True)
            reward = result.reward
            norm_reward = result.normalized_reward
            prob_reward = result.probabilistic_reward
            norm_prob_reward = result.normalized_probabilistic_reward

            print("  action={}, reward={}, prob_reward={}".format(action, reward, prob_reward))

            rewards[action] = reward
            norm_rewards[action] = norm_reward
            prob_rewards[action] = prob_reward
            norm_prob_rewards[action] = norm_prob_reward
            assert(prob_reward >= 0)
            results.append(result)

        # Keep record for saving later
        record = data_record.RecordV2(obs_levels, grid_3ds, rewards, norm_rewards,
                                      prob_rewards, norm_prob_rewards, scores)
        records.append(record)

        # Perform epsilon-greedy action.
        if np.random.rand() < epsilon:
            action = np.random.randint(0, environment.get_num_of_actions())
        else:
            max_prob_reward = np.max(prob_rewards)
            actions = np.arange(environment.get_num_of_actions())[prob_rewards == max_prob_reward]
            if len(actions) == 1:
                action = actions[0]
            else:
                # If there is not a single best action, redo the previous one if it is one of the best.
                if prev_action in actions:
                    action = prev_action
                else:
                    action = np.random.choice(actions)

        # TODO: When getting the depth image again here, sometimes it differs
        #       from the depth image in the loop above (i.e. leading to a different reward).
        #       It's not clear why this happens. Simple tests on Unreal Engine and UnrealCV
        #       indicate that the depth maps do not change.
        # new_pose = environment.simulate_action_on_pose(current_pose, action)
        # environment.set_pose(new_pose, wait_until_set=True)
        # depth_image = environment.get_engine().get_depth_image()
        # result = environment.get_mapper().perform_insert_depth_map_rpy(
        #     new_pose.location(), new_pose.orientation_rpy(),
        #     depth_image, intrinsics, downsample_to_grid=True, simulate=False)
        #
        # if not np.all(depth_image == depth_images[action]):
        #     diff_image = np.abs(depth_image - depth_images[action])
        #     print(np.sum(diff_image > 0))
        #     print(np.max(diff_image))
        #     max_depth = np.max(depth_image[diff_image > 0])
        #     import cv2
        #     cv2.imshow("depth_image1", depth_image / max_depth)
        #     cv2.imshow("depth_image2", depth_images[action] / max_depth)
        #     cv2.imshow("diff_image", diff_image / np.max(diff_image))
        #     cv2.waitKey(0)
        # assert(np.all(depth_image == depth_images[action]))

        environment.set_pose(new_poses[action], wait_until_set=True)
        result = environment.get_mapper().perform_insert_depth_map_rpy(
            new_poses[action].location(), new_poses[action].orientation_rpy(),
            depth_images[action], intrinsics, downsample_to_grid=True, simulate=False)
        print("Selected action: {}".format(action))
        assert(prob_rewards[action] == result.probabilistic_reward)
        # print("score={}".format(result.score))
        # print("probabilistic_score={}".format(result.probabilistic_score))
        # print("normalized_score={}".format(result.normalized_score))
        # print("normalized_probabilistic_score={}".format(result.normalized_probabilistic_score))
        assert(record.rewards[action] == result.reward)

        # print("action={}, reward={}".format(action, reward))

        prev_action = action

        if not args.dry_run and len(records) % records_per_file == 0:
            # filename, current_file_num = get_next_output_tf_filename(current_file_num)
            filename, current_file_num = file_helpers.get_next_output_hdf5_filename(
                current_file_num, template=filename_template)
            print("Writing records to file {}".format(filename))
            # write_tf_records(filename, records)
            data_record.write_hdf5_records_v2(filename, records)
            if check_written_records:
                print("Reading records from file {}".format(filename))
                records_read = data_record.read_hdf5_records_v2_as_list(filename)
                for record, record_read in zip(records, records_read):
                    assert(np.all(record.obs_levels == record_read.obs_levels))
                    for grid_3d, grid_3d_read in zip(record.grid_3d, record_read.grid_3d):
                        assert(np.all(grid_3d == grid_3d_read))
                    assert(np.all(record.rewards == record_read.rewards))
            records = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--dry-run', action='store_true', help="Do not save anything")
    parser.add_argument('--output-path', type=str, help="Output path")
    parser.add_argument('--environment', type=str, help="Environment name")
    parser.add_argument('--num-records', default=10000, type=int, help="Total records to collect")
    parser.add_argument('--client-id', default=0, type=int)

    args = parser.parse_args()

    run(args)
