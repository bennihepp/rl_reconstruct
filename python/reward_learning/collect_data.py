#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import numpy as np
import data_record
import env_factory
import file_helpers
import RLrecon.environments.environment as RLenvironment


def run(args):
    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    filename_template = os.path.join(output_path, file_helpers.DEFAULT_HDF5_TEMPLATE)

    num_records = args.num_records
    records_per_file = 1000
    reset_interval = 100
    reset_score_threshold = 0.65
    check_written_records = True

    obs_levels = [0, 1, 2, 3]
    obs_sizes_x = [16] * len(obs_levels)
    obs_sizes_y = obs_sizes_x
    obs_sizes_z = obs_sizes_x

    # environment_class = RLenvironment.HorizontalEnvironment
    environment_class = RLenvironment.SimpleV0Environment
    # environment_class = RLenvironment.SimpleV2Environment
    client_id = args.client_id
    environment = env_factory.create_environment(environment_class, client_id)

    intrinsics = np.zeros((3, 3))
    intrinsics[0, 0] = environment.get_engine().get_focal_length()
    intrinsics[1, 1] = environment.get_engine().get_focal_length()
    intrinsics[0, 2] = environment.get_engine().get_width() / 2
    intrinsics[1, 2] = environment.get_engine().get_height() / 2

    current_file_num = 0
    score = 0.0
    records = []
    environment.reset()
    for i in xrange(num_records):
        print("Record #{}".format(i))
        if i % reset_interval == 0 or score >= reset_score_threshold:
            environment.reset()
        current_pose = environment.get_pose()

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
            grid_3d = np.stack([occupancies_3d, observation_counts_3d], axis=-1)
            if grid_3ds is None:
                grid_3ds = grid_3d
            else:
                grid_3ds = np.concatenate([grid_3ds, grid_3d], axis=-1)

        result = environment.get_mapper().perform_info()
        score = result.score
        normalized_score = result.normalized_score
        prob_score = result.probabilistic_score
        normalized_prob_score = result.normalized_probabilistic_score
        scores = np.array([score, normalized_score, prob_score, normalized_prob_score])

        results = []
        rewards = np.zeros((environment.get_num_of_actions(),))
        norm_rewards = np.zeros((environment.get_num_of_actions(),))
        prob_rewards = np.zeros((environment.get_num_of_actions(),))
        norm_prob_rewards = np.zeros((environment.get_num_of_actions(),))
        for action in xrange(environment.get_num_of_actions()):
            new_pose = environment.simulate_action_on_pose(current_pose, action)
            environment.set_pose(new_pose, wait_until_set=True)
            # point_cloud = environment._get_depth_point_cloud(new_pose)
            # result = environment.get_mapper().perform_insert_point_cloud_rpy(new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
            depth_image = environment.get_engine().get_depth_image()
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
        record = data_record.RecordV2(obs_levels, grid_3ds, rewards, norm_rewards,
                                      prob_rewards, norm_prob_rewards, scores)
        records.append(record)

        # Perform random action
        action = np.random.randint(0, environment.get_num_of_actions())
        new_pose = environment.simulate_action_on_pose(current_pose, action)
        environment.set_pose(new_pose, wait_until_set=True)
        # Sanity check of rewards
        # point_cloud = environment._get_depth_point_cloud(new_pose)
        # result = environment.get_mapper().perform_insert_point_cloud_rpy(new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=False)
        # result = environment.get_mapper().perform_insert_point_cloud_rpy(new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
        # result2 = environment.get_mapper().perform_insert_point_cloud_rpy(new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
        # result2 = environment.get_mapper().perform_insert_point_cloud_rpy(new_pose.location(), new_pose.orientation_rpy(), point_cloud)
        depth_image = environment.get_engine().get_depth_image()
        # import cv2
        # cv2.imshow("depth", depth_image / 10.)
        # cv2.waitKey(10)
        # and compute its reward
        result = environment.get_mapper().perform_insert_depth_map_rpy(
            new_pose.location(), new_pose.orientation_rpy(),
            depth_image, intrinsics, downsample_to_grid=True, simulate=False)
        score = result.normalized_score
        # score = result2.normalized_score
        if prob_rewards[action] != result.probabilistic_reward:
            print(rewards[action], result.reward)
            print(prob_rewards[action], result.probabilistic_reward)
            print(results[action].score, result.score)
            print(results[action].probabilistic_score, result.probabilistic_score)
        assert(prob_rewards[action] == result.probabilistic_reward)
        # print("score={}".format(result.score))
        # print("probabilistic_score={}".format(result.probabilistic_score))
        # print("normalized_score={}".format(result.normalized_score))
        # print("normalized_probabilistic_score={}".format(result.normalized_probabilistic_score))
        if record.rewards[action] != result.reward:
            print(record.rewards[action], result.reward)
            print(record.rewards[action], result.probabilistic_reward)
        assert(record.rewards[action] == result.reward)

        # print("action={}, reward={}".format(action, reward))

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
    parser.add_argument('--dry-run', action='store_true', help='Do not save anything')
    parser.add_argument('--output-path', help='Output path')
    parser.add_argument('--num-records', default=10000, type=int, help='Total records to collect')
    parser.add_argument('--client-id', default=0, type=int)

    args = parser.parse_args()

    run(args)
