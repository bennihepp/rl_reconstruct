#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from builtins import range

import os
import argparse
import numpy as np
import uuid
import yaml
from pybh import hdf5_utils
import env_factory
import file_helpers
import data_record
from pybh.utils import Timer, TimeMeter, DummyTimeMeter, argparse_bool
from policy_helpers import VisitedPoses
from pybh import math_utils


def plot_grid(occupancies_3d, observation_certainties_3d):
    import visualize_data
    import visualization
    import mayavi.mlab as mlab
    x, y, z = visualize_data.get_xyz_grids(occupancies_3d, scale=0.1)

    s = occupancies_3d
    fig = visualize_data.create_mlab_figure()
    visualize_data.plot3d_with_threshold(x, y, z, s, low_thres=0.2, fig=fig)
    mlab.title("Occupancy")
    mlab.scalarbar()

    s = occupancies_3d
    fig = visualize_data.create_mlab_figure()
    visualize_data.plot3d_with_threshold(x, y, z, s, fig=fig)
    mlab.title("Occupancy")
    mlab.scalarbar()

    s = observation_certainties_3d
    fig = visualize_data.create_mlab_figure()
    visualize_data.plot3d_with_threshold(x, y, z, s, high_thres=0.5, fig=fig)
    mlab.title("Observation count")
    mlab.scalarbar()

    s = observation_certainties_3d
    fig = visualize_data.create_mlab_figure()
    visualize_data.plot3d_with_threshold(x, y, z, s, fig=fig)
    mlab.title("Observation count2")
    mlab.scalarbar()

    mlab.show()


def query_octomap(environment, pose, obs_levels, obs_sizes, map_resolution, axis_mode=0, forward_factor=3 / 8.):
    grid_3ds = None
    for k in range(len(obs_levels)):
        obs_level = obs_levels[k]
        obs_size_x = obs_sizes[0]
        obs_size_y = obs_sizes[1]
        obs_size_z = obs_sizes[2]

        obs_resolution = map_resolution * (2 ** obs_level)
        offset_x = obs_resolution * obs_size_x * forward_factor
        offset_vec = math_utils.rotate_vector_with_rpy(pose.orientation_rpy(), [offset_x, 0, 0])
        query_location = pose.location() + offset_vec
        query_pose = environment.Pose(query_location, pose.orientation_rpy())

        res = environment.get_mapper().perform_query_subvolume_rpy(
            query_pose.location(), query_pose.orientation_rpy(),
            obs_level, obs_size_x, obs_size_y, obs_size_z, axis_mode)
        occupancies = np.asarray(res.occupancies, dtype=np.float32)
        occupancies_3d = np.reshape(occupancies, (obs_size_x, obs_size_y, obs_size_z))
        observation_certainties = np.asarray(res.observation_certainties, dtype=np.float32)
        observation_certainties_3d = np.reshape(observation_certainties, (obs_size_x, obs_size_y, obs_size_z))
        # Plot histograms
        # num_bins = 50
        # plt.figure()
        # plt.hist(occupancies_3d.flatten(), num_bins)
        # plt.title("occupancies level {}".format(obs_level))
        # plt.figure()
        # plt.hist(observation_certainties.flatten(), num_bins)
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
        #     np.mean(observation_certainties.flatten()),
        #     np.std(observation_certainties.flatten()),
        #     np.min(observation_certainties.flatten()),
        #     np.max(observation_certainties.flatten()),
        # ))
        grid_3d = np.stack([occupancies_3d, observation_certainties_3d], axis=-1)
        if grid_3ds is None:
            grid_3ds = grid_3d
        else:
            grid_3ds = np.concatenate([grid_3ds, grid_3d], axis=-1)
    # plt.show()
    return grid_3ds


def run(args):
    wait_until_pose_set = args.wait_until_pose_set
    measure_timing = args.measure_timing
    if measure_timing:
        time_meter = TimeMeter()
    else:
        time_meter = DummyTimeMeter()

    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    filename_template = os.path.join(output_path, file_helpers.DEFAULT_HDF5_TEMPLATE)

    samples_per_file = args.samples_per_file
    num_files = args.num_files
    num_samples = num_files * samples_per_file
    reset_interval = args.reset_interval
    reset_score_threshold = args.reset_score_threshold
    check_written_samples = args.check_written_samples

    dataset_kwargs = {}
    if args.compression:
        dataset_kwargs.update({"compression": args.compression})
        if args.compression_level >= 0:
            dataset_kwargs.update({"compression_opts": args.compression_level})

    epsilon = args.epsilon

    obs_levels = [int(x.strip()) for x in args.obs_levels.strip("[]").split(",")]
    obs_sizes = [int(x.strip()) for x in args.obs_sizes.strip("[]").split(",")]
    print("obs_levels={}".format(obs_levels))
    print("obs_sizes={}".format(obs_sizes))

    client_id = args.client_id
    with open(args.environment_config, "r") as fin:
        environment_config = yaml.load(fin)
    environment = env_factory.create_environment_from_config(environment_config, client_id)

    intrinsics = environment.get_engine().get_intrinsics()
    result = environment.get_mapper().perform_info()
    map_resolution = result.resolution
    axis_mode = environment_config["collect_data"]["axis_mode"]
    forward_factor = environment_config["collect_data"]["forward_factor"]
    downsample_to_grid = environment_config["collect_data"]["downsample_to_grid"]
    print("map_resolution={}".format(map_resolution))
    print("axis_mode={}".format(axis_mode))
    print("forward_factor={}".format(forward_factor))
    print("downsample_to_grid={}".format(downsample_to_grid))

    if args.manual:
        environment.get_engine().enable_input()

        import time
        environment.reset(keep_pose=True)
        while True:
            current_pose = environment.get_pose()

            rgb_image, depth_image, normal_image = environment.get_engine().get_rgb_depth_normal_images(use_trackball=True)
            rgb_image = np.asarray(rgb_image, dtype=np.float32)
            depth_image = np.asarray(depth_image, dtype=np.float32)
            normal_image = np.asarray(normal_image, dtype=np.float32)

            if args.visualize:
                if depth_image.shape[0] == 1:
                    import matplotlib.pyplot as plt
                    fig = plt.figure(1)
                    plt.clf()
                    plt.step(np.arange(depth_image.shape[1]), depth_image[0, ...])
                    plt.title("Depth image")
                    fig.canvas.draw()
                    plt.show(block=False)
                else:
                    import cv2
                    cv2.imshow("depth_image", depth_image / np.max(depth_image))
                    cv2.waitKey(50)

                    import matplotlib.pyplot as plt
                    fig = plt.figure(1)
                    plt.clf()
                    i = depth_image.shape[0] / 2
                    print(i)
                    i = int(i)
                    plt.step(np.arange(depth_image.shape[1]), depth_image[i, ...])
                    plt.title("Depth image")
                    fig.canvas.draw()
                    plt.show(block=False)

            # Query octomap
            print("current_pose: {}".format(current_pose.orientation_rpy()))
            in_grid_3ds = query_octomap(environment, current_pose, obs_levels, obs_sizes,
                                        map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
            in_grid_3ds = np.asarray(in_grid_3ds, dtype=np.float32)
            if args.visualize:
                fig = 1
                import visualization
                visualization.clear_figure(fig)
                visualization.plot_grid(in_grid_3ds[..., 0], in_grid_3ds[..., 1], title_prefix="input", show=False, fig_offset=fig)
                visualization.show(stop=True)

            environment.get_mapper().perform_insert_depth_map_rpy(
                current_pose.location(), current_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)

            time.sleep(0.5)
        return

    # environment.get_engine().test()
    environment.get_engine().disable_input()

    def read_samples_from_file(filename):
        stacked_samples, attr_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(filename, read_attributes=True)
        samples = []
        for key in stacked_samples:
            for i in range(len(stacked_samples[key])):
                if len(samples) <= i:
                    samples.append({})
                samples[i][key] = stacked_samples[key][i, ...]
        return samples, attr_dict

    def write_samples_to_next_file(samples, attr_dict, next_file_num):
        filename, next_file_num = file_helpers.get_next_output_hdf5_filename(
            next_file_num, template=filename_template)
        print("Writing samples to file {}".format(filename))
        if not args.dry_run:
            data_record.write_samples_to_hdf5_file(filename, samples, attr_dict, **dataset_kwargs)
            if check_written_samples:
                print("Reading samples from file {}".format(filename))
                samples_read, attr_dict_read = read_samples_from_file(filename)
                assert(len(samples) == len(samples_read))
                for i in range(len(samples)):
                    for key in samples[i]:
                        assert(np.all(samples[i][key] == samples_read[i][key]))
                for key in attr_dict:
                    assert(np.all(attr_dict[key] == attr_dict_read[key]))
        return next_file_num

    next_file_num = 0
    samples = []
    attr_dict = None
    prev_action = np.random.randint(0, environment.get_num_of_actions())
    normalized_prob_score = 0
    # Make sure we reset the environment at the start
    reset_env = True
    total_steps = -1
    while True:
        total_steps += 1
        print(next_file_num, num_files, total_steps, num_samples)
        if next_file_num >= num_files and total_steps >= num_samples:
            break

        i = total_steps

        # Get current normalized prob score to check termination
        tmp_result = environment.get_mapper().perform_info()
        normalized_prob_score = tmp_result.normalized_probabilistic_score

        if not args.keep_episodes_together and len(samples) >= samples_per_file:
            next_file_num = write_samples_to_next_file(samples, attr_dict, next_file_num)
            samples = []
            attr_dict = None

        if reset_env or \
                (i % reset_interval == 0) \
                or normalized_prob_score >= reset_score_threshold:
            print("Resetting environment")
            if len(samples) >= samples_per_file:
                print("Writing {} recorded samples to disk".format(len(samples)))
                next_file_num = write_samples_to_next_file(samples, attr_dict, next_file_num)
                samples = []
                attr_dict = None
            sample_id = 0
            reset_env = False
            environment.reset()
            visited_poses = VisitedPoses(3 + 4, np.concatenate([np.ones((3,)), 10. * np.ones((4,))]))
            episode_uuid = uuid.uuid1()
            episode_id = np.fromstring(episode_uuid.bytes, dtype=np.uint8)

            if args.collect_center_grid_of_previous_pose:
                center_grid_of_previous_pose = None

        print("Total step #{}, episode step #{}, # of samples {}".format(i, sample_id, len(samples)))
        current_pose = environment.get_pose()

        # Get current scores
        with time_meter.measure("get_info"):
            result = environment.get_mapper().perform_info()
            scores = np.asarray([result.score, result.normalized_score,
                               result.probabilistic_score, result.normalized_probabilistic_score], dtype=np.float32)

        print("  scores: {}".format(scores))

        visited_poses.increase_visited_pose(current_pose)

        # Simulate effect of actions and compute depth maps and rewards
        prob_rewards = np.zeros((environment.get_num_of_actions(),))
        visit_counts = np.zeros((environment.get_num_of_actions(),))
        collision_flags = np.zeros((environment.get_num_of_actions(),))
        if not args.collect_only_selected_action:
            in_grid_3ds_array = [None] * environment.get_num_of_actions()
            out_grid_3ds_array = [None] * environment.get_num_of_actions()
            result_array = [None] * environment.get_num_of_actions()
            rgb_images = [None] * environment.get_num_of_actions()
            depth_images = [None] * environment.get_num_of_actions()
            normal_images = [None] * environment.get_num_of_actions()
        for action in range(environment.get_num_of_actions()):
            with time_meter.measure("simulate_collision"):
                colliding = environment.is_action_colliding(current_pose, action)
            if colliding:
                print("Action {} would collide".format(action))
                collision_flags[action] = 1
                continue
            new_pose = environment.simulate_action_on_pose(current_pose, action)

            if not args.collect_only_selected_action:
                with time_meter.measure("simulate_query_octomap"):
                    in_grid_3ds_array[action] = query_octomap(
                        environment, new_pose, obs_levels, obs_sizes,
                        map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
                    in_grid_3ds_array[action] = np.asarray(in_grid_3ds_array[action], dtype=np.float32)

            with time_meter.measure("simulate_set_pose"):
                environment.set_pose(new_pose, wait_until_set=wait_until_pose_set, broadcast=False)
            new_pose_retrieved = environment.get_pose()
            assert np.allclose(new_pose_retrieved.location(), new_pose.location())
            assert np.allclose(new_pose_retrieved.orientation_rpy(), new_pose.orientation_rpy())

            # point_cloud = environment._get_depth_point_cloud(new_pose)
            # result = environment.get_mapper().perform_insert_point_cloud_rpy(
            #     new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
            with time_meter.measure("simulate_image_retrieval"):
                if args.collect_only_depth_image or args.collect_only_selected_action or args.collect_no_images:
                    depth_image = environment.get_engine().get_depth_image()
                    depth_image = np.asarray(depth_image, dtype=np.float32)

                else:
                    rgb_image, depth_image, normal_image = environment.get_engine().get_rgb_depth_normal_images()
                    rgb_image = np.asarray(rgb_image, dtype=np.float32)
                    depth_image = np.asarray(depth_image, dtype=np.float32)
                    normal_image = np.asarray(normal_image, dtype=np.float32)
                if not args.collect_only_selected_action:
                    depth_images[action] = depth_image
                    if not args.collect_only_depth_image:
                        rgb_images[action] = rgb_image
                        normal_images[action] = normal_image

            simulate = True
            if not args.collect_only_selected_action and args.collect_output_grid:
                simulate = False
                with time_meter.measure("push_octomap"):
                    environment.get_mapper().perform_push_octomap()
            with time_meter.measure("simulate_insert_depth_image"):
                result = environment.get_mapper().perform_insert_depth_map_rpy(
                    new_pose.location(), new_pose.orientation_rpy(),
                    depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=simulate)

            if not args.collect_only_selected_action:
                result_array[action] = result
            prob_reward = result.probabilistic_reward

            prob_rewards[action] = prob_reward
            assert(prob_reward >= 0)

            visit_count = visited_poses.get_visit_count(new_pose)
            visit_counts[action] = visit_count

            if not args.collect_only_selected_action and args.collect_output_grid:
                with time_meter.measure("simulate_query_octomap"):
                    out_grid_3ds_array[action] = query_octomap(
                        environment, new_pose, obs_levels, obs_sizes,
                        map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
                    out_grid_3ds_array[action] = np.asarray(out_grid_3ds_array[action], dtype=np.float32)
                with time_meter.measure("pop_octomap"):
                    environment.get_mapper().perform_pop_octomap()

        print("Possible rewards: {}".format(prob_rewards))

        with time_meter.measure("select_action"):
            visit_counts = np.array(visit_counts, dtype=np.float32)
            visit_weights = 1. / (visit_counts + 1)
            adjusted_rewards = prob_rewards * visit_weights
            adjusted_rewards[collision_flags > 0] = - np.finfo(np.float32).max
            print("Adjusted expected rewards:", adjusted_rewards)

            if np.all(collision_flags[action] > 0):
                reset_env = True
                continue

            # Perform epsilon-greedy action.
            if np.random.rand() < epsilon:
                valid_action_indices = np.arange(environment.get_num_of_actions())
                valid_action_indices = valid_action_indices[collision_flags == 0]
                assert(len(valid_action_indices) > 0)
                selected_action = np.random.choice(valid_action_indices)
            else:
                max_prob_reward = np.max(adjusted_rewards)
                actions = np.arange(environment.get_num_of_actions())[adjusted_rewards == max_prob_reward]
                if len(actions) == 1:
                    selected_action = actions[0]
                else:
                    # If there is not a single best action, redo the previous one if it is one of the best.
                    if prev_action in actions:
                        selected_action = prev_action
                    else:
                        selected_action = np.random.choice(actions)
            # print("Selected action: {}".format(action))

            if np.all(collision_flags[selected_action] > 0):
                reset_env = True
                continue

        new_pose = environment.simulate_action_on_pose(current_pose, selected_action)
        with time_meter.measure("set_pose"):
            environment.set_pose(new_pose, wait_until_set=wait_until_pose_set)

        with time_meter.measure("image_retrieval"):
            if args.collect_only_depth_image:
                depth_image = environment.get_engine().get_depth_image()
                depth_image = np.asarray(depth_image, dtype=np.float32)
            else:
                rgb_image, depth_image, normal_image = environment.get_engine().get_rgb_depth_normal_images()
                rgb_image = np.asarray(rgb_image, dtype=np.float32)
                depth_image = np.asarray(depth_image, dtype=np.float32)
                normal_image = np.asarray(normal_image, dtype=np.float32)

        if args.visualize:
            if depth_image.shape[0] == 1:
                import matplotlib.pyplot as plt
                fig = plt.figure(1)
                plt.clf()
                # plt.plot(np.arange(depth_image.shape[1]), depth_image[0, ...])
                plt.step(np.arange(depth_image.shape[1]), depth_image[0, ...])
                plt.title("Depth image")
                fig.canvas.draw()
                plt.show(block=False)
            else:
                import cv2
                cv2.imshow("depth_image", depth_image / np.max(depth_image))
                cv2.waitKey(50)

        # Query octomap
        if args.collect_only_selected_action:
            with time_meter.measure("query_octomap"):
                in_grid_3ds = query_octomap(environment, new_pose, obs_levels, obs_sizes,
                                            map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
                in_grid_3ds = np.asarray(in_grid_3ds, dtype=np.float32)

        if args.collect_center_grid_of_previous_pose:
            with time_meter.measure("query_octomap"):
                center_in_grid_3ds = query_octomap(environment, current_pose, obs_levels, obs_sizes,
                                                   map_resolution, axis_mode=axis_mode, forward_factor=0.0)
                center_in_grid_3ds = np.asarray(center_in_grid_3ds, dtype=np.float32)

        if args.visualize:
            fig = 1
            import visualization
            fig = visualization.plot_grid(in_grid_3ds[..., 2], in_grid_3ds[..., 3], title_prefix="input_1", show=False, fig_offset=fig)
            visualization.clear_figure(fig)
            visualization.plot_grid(in_grid_3ds[..., 4], in_grid_3ds[..., 5], title_prefix="input_2", show=False, fig_offset=fig)
            visualization.show(stop=True)

        # sim_result = environment.get_mapper().perform_insert_depth_map_rpy(
        #     new_pose.location(), new_pose.orientation_rpy(),
        #     depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=True)
        with time_meter.measure("insert_depth_image"):
            result = environment.get_mapper().perform_insert_depth_map_rpy(
                new_pose.location(), new_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)
        # print("result diff:", sim_result.probabilistic_reward - result.probabilistic_reward)
        # assert(sim_result.probabilistic_reward - result.probabilistic_reward == 0)

        print("Selected action={}, probabilistic reward={}".format(selected_action, result.probabilistic_reward))
        if args.collect_output_grid:
            print("Grid differences:", [np.sum(out_grid_3ds[..., i] - in_grid_3ds[..., i]) for i in range(in_grid_3ds.shape[-1])])

        if attr_dict is None:
            attr_dict = {}
            attr_dict["intrinsics"] = intrinsics
            attr_dict["map_resolution"] = map_resolution
            attr_dict["axis_mode"] = axis_mode
            attr_dict["forward_factor"] = forward_factor
            attr_dict["obs_levels"] = obs_levels

        if args.collect_center_grid_of_previous_pose and center_grid_of_previous_pose is None:
            skip_sample = True
        else:
            skip_sample = False

        # Create and keep samples for saving later
        if (not skip_sample) and args.collect_only_selected_action:
            if args.collect_output_grid:
                # Query octomap
                with time_meter.measure("query_octomap"):
                    out_grid_3ds = query_octomap(environment, new_pose, obs_levels, obs_sizes,
                                                 map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
                    out_grid_3ds = np.asarray(out_grid_3ds, dtype=np.float32)

            rewards = np.asarray([result.reward, result.normalized_reward,
                                  result.probabilistic_reward, result.normalized_probabilistic_reward], dtype=np.float32)
            new_scores = np.asarray([result.score, result.normalized_score,
                                     result.probabilistic_score, result.normalized_probabilistic_score], dtype=np.float32)

            sample = {"in_grid_3ds": in_grid_3ds,
                      "rewards": rewards,
                      "scores": scores,
                      "new_scores": new_scores,
                      "episode_id": episode_id,
                      "sample_id": np.array(sample_id, dtype=np.int32),
                      "selected_action": np.array(True, dtype=np.int8),
                      "action_index": np.array(selected_action, dtype=np.int8)}
            if not args.collect_no_images:
                sample["depth_image"] = depth_image
            if args.collect_output_grid:
                sample["out_grid_3ds"] = out_grid_3ds
            if not args.collect_only_depth_image:
                sample["rgb_image"] = rgb_image
                sample["normal_image"] = normal_image
            if args.collect_center_grid_of_previous_pose:
                sample["center_in_grid_3ds"] = center_in_grid_3ds
            samples.append(sample)
        elif not skip_sample:
            for action in range(environment.get_num_of_actions()):
                result = result_array[action]
                if result is None:
                    continue
                rewards = np.asarray([result.reward, result.normalized_reward,
                                    result.probabilistic_reward, result.normalized_probabilistic_reward], dtype=np.float32)
                new_scores = np.asarray([result.score, result.normalized_score,
                                   result.probabilistic_score, result.normalized_probabilistic_score], dtype=np.float32)
                in_grid_3ds = in_grid_3ds_array[action]
                assert(in_grid_3ds is not None)
                sample = {"in_grid_3ds": in_grid_3ds,
                          "rewards": rewards,
                          "scores": scores,
                          "new_scores": new_scores,
                          "episode_id": episode_id,
                          "sample_id": np.array(sample_id, dtype=np.int32),
                          "selected_action": np.array(action == selected_action, dtype=np.int8),
                          "action_index": np.array(selected_action, dtype=np.int8)}
                if not args.collect_no_images:
                    assert(depth_images[action] is not None)
                    sample["depth_image"] = depth_images[action]
                if args.collect_output_grid:
                    assert(out_grid_3ds_array[action] is not None)
                    sample["out_grid_3ds"] = out_grid_3ds_array[action]
                if not args.collect_only_depth_image:
                    assert(rgb_images[action] is not None)
                    assert(normal_images[action] is not None)
                    sample["rgb_image"] = rgb_image
                    sample["normal_image"] = normal_image
                if args.collect_center_grid_of_previous_pose:
                    sample["center_in_grid_3ds"] = center_in_grid_3ds
                samples.append(sample)

        sample_id += 1
        prev_action = selected_action

        time_meter.print_times()

    if len(samples) > 0:
        write_samples_to_next_file(samples, attr_dict, next_file_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--dry-run', type=argparse_bool, default=False, help="Do not save anything")
    parser.add_argument('--output-path', required=True, type=str, help="Output path")
    parser.add_argument('--obs-levels', default="0,1,2,3,4", type=str)
    parser.add_argument('--obs-sizes', default="16,16,16", type=str)
    parser.add_argument('--samples-per-file', default=1000, type=int, help="Samples per file")
    parser.add_argument('--num-files', default=100, type=int, help="Number of files")
    parser.add_argument('--environment-config', type=str, required=True, help="Environment configuration file")
    parser.add_argument('--reset-interval', default=100, type=int)
    parser.add_argument('--reset-score-threshold', default=0.3, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)
    parser.add_argument('--collect-center-grid-of-previous-pose', type=argparse_bool, default=False)
    parser.add_argument('--collect-only-selected-action', type=argparse_bool, default=False)
    parser.add_argument('--collect-only-depth-image', type=argparse_bool, default=True)
    parser.add_argument('--collect-output-grid', type=argparse_bool, default=False)
    parser.add_argument('--collect-no-images', type=argparse_bool, default=False)
    parser.add_argument('--keep-episodes-together', type=argparse_bool, default=False)
    parser.add_argument('--client-id', default=0, type=int)
    parser.add_argument('--wait-until-pose-set', type=argparse_bool, default=True,
                        help="Wait until pose is set in Unreal Engine")
    parser.add_argument('--measure-timing', type=argparse_bool, default=False, help="Measure timing of steps")
    parser.add_argument('--visualize', type=argparse_bool, default=False)
    parser.add_argument('--compression', type=str, default="gzip", help="Type of compression to use. Default is gzip.")
    parser.add_argument('--compression-level', default=5, type=int, help="Gzip compression level")
    parser.add_argument('--check-written-samples', type=argparse_bool, default=True,
                        help="Whether written files should be read and checked afterwards.")

    args = parser.parse_args()

    run(args)
