#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
import yaml
import data_record
import env_factory
import file_helpers
from pybh.utils import Timer, DummyTimingMeasurer, TimingMeasurer, argparse_bool
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
    for k in xrange(len(obs_levels)):
        obs_level = obs_levels[k]
        obs_size_x = obs_sizes[k][0]
        obs_size_y = obs_sizes[k][1]
        obs_size_z = obs_sizes[k][2]

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
    timer = Timer()
    timer2 = Timer()
    wait_until_pose_set = args.wait_until_pose_set

    if args.measure_timing:
        timing_measurer = TimingMeasurer()
    else:
        timing_measurer = DummyTimingMeasurer()

    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    filename_template = os.path.join(output_path, file_helpers.DEFAULT_HDF5_TEMPLATE)

    records_per_file = args.records_per_file
    num_files = args.num_files
    num_records = num_files * records_per_file
    reset_interval = args.reset_interval
    reset_score_threshold = args.reset_score_threshold
    check_written_records = args.check_written_records

    dataset_kwargs = {}
    if args.compression_level >= 0:
        dataset_kwargs.update({"compression": "gzip",
                               "compression_opts": args.compression_level})

    epsilon = args.epsilon

    obs_levels = [int(x.strip()) for x in args.obs_levels.strip("[]").split(",")]
    obs_sizes = [int(x.strip()) for x in args.obs_sizes.strip("[]").split(",")]
    obs_sizes = [obs_sizes] * len(obs_levels)
    print("obs_levels={}".format(obs_levels))
    print("obs_sizes={}".format(obs_sizes))

    client_id = args.client_id
    with file(args.environment_config, "r") as fin:
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

            rgb_image, depth_image, normal_image = environment.get_engine().get_rgb_depth_normal_images()
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
            if args.visualize:
                fig = 1
                import visualization
                visualization.clear_figure(fig)
                visualization.plot_grid(in_grid_3ds[..., 0], in_grid_3ds[..., 1], title_prefix="input", show=False, fig_offset=fig)
                visualization.show(stop=True)

            result = environment.get_mapper().perform_insert_depth_map_rpy(
                current_pose.location(), current_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)

            time.sleep(0.5)
        return

    # environment.get_engine().test()
    environment.get_engine().disable_input()

    next_file_num = 0
    records = []
    # environment.reset()
    environment.get_mapper().perform_load_surface_voxels()
    prev_action = np.random.randint(0, environment.get_num_of_actions())
    reset_env = False
    for i in xrange(num_records):
        if next_file_num >= num_files:
            break

        print("Record #{}".format(i))
        current_pose = environment.get_pose()

        result = environment.get_mapper().perform_info()
        score = result.score
        normalized_score = result.normalized_score
        prob_score = result.probabilistic_score
        normalized_prob_score = result.normalized_probabilistic_score
        scores = np.array([score, normalized_score, prob_score, normalized_prob_score])

        print("  scores: {}".format(scores))

        if reset_env or \
                (i % reset_interval == 0) \
                or normalized_prob_score >= reset_score_threshold:
            print("Resetting environment")
            reset_env = False
            environment.reset()
            visited_poses = VisitedPoses(3 + 4, np.concatenate([np.ones((3,)), 10. * np.ones((4,))]))

        visited_poses.increase_visited_pose(current_pose)

        if measure_timing:
            timer2.restart()

        # Simulate effect of actions and compute depth maps and rewards
        prob_rewards = np.zeros((environment.get_num_of_actions(),))
        visit_counts = np.zeros((environment.get_num_of_actions(),))
        collision_flags = np.zeros((environment.get_num_of_actions(),))
        for action in xrange(environment.get_num_of_actions()):
            with timing_measurer.measurement("simulate_collision"):
                colliding = environment.is_action_colliding(current_pose, action)
            if colliding:
                print("Action {} would collide".format(action))
                collision_flags[action] = 1
                continue
            with timing_measurer.measurement("simulate_action"):
                new_pose = environment.simulate_action_on_pose(current_pose, action)
            with timing_measurer.measurement("simulate_set_pose"):
                environment.set_pose(new_pose, wait_until_set=wait_until_pose_set)
            # point_cloud = environment._get_depth_point_cloud(new_pose)
            # result = environment.get_mapper().perform_insert_point_cloud_rpy(
            #     new_pose.location(), new_pose.orientation_rpy(), point_cloud, simulate=True)
            with timing_measurer.measurement("simulate_get_depth_image"):
                depth_image = environment.get_engine().get_depth_image()
            with timing_measurer.measurement("simulate_insert_depth_image"):
                result = environment.get_mapper().perform_insert_depth_map_rpy(
                    new_pose.location(), new_pose.orientation_rpy(),
                    depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=True)
            prob_reward = result.probabilistic_reward

            prob_rewards[action] = prob_reward
            assert(prob_reward >= 0)

            visit_count = visited_poses.get_visit_count(new_pose)
            visit_counts[action] = visit_count

        with timing_measurer.measurement("select_action"):
            print("Possible rewards: {}".format(prob_rewards))

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
                action = np.random.choice(valid_action_indices)
            else:
                max_prob_reward = np.max(adjusted_rewards)
                actions = np.arange(environment.get_num_of_actions())[adjusted_rewards == max_prob_reward]
                if len(actions) == 1:
                    action = actions[0]
                else:
                    # If there is not a single best action, redo the previous one if it is one of the best.
                    if prev_action in actions:
                        action = prev_action
                    else:
                        action = np.random.choice(actions)
            # print("Selected action: {}".format(action))

            if np.all(collision_flags[action] > 0):
                reset_env = True
                continue

        with timing_measurer.measurement("simulate_action"):
            new_pose = environment.simulate_action_on_pose(current_pose, action)
        with timing_measurer.measurement("set_pose"):
            environment.set_pose(new_pose, wait_until_set=wait_until_pose_set)

        # Get current scores
        with timing_measurer.measurement("get_info"):
            result = environment.get_mapper().perform_info()
            scores = np.asarray([result.score, result.normalized_score,
                               result.probabilistic_score, result.normalized_probabilistic_score], dtype=np.float32)

        with timing_measurer.measurement("simulate_rgb_depth_normal_images"):
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
        with timing_measurer.measurement("query_octomap"):
            in_grid_3ds = query_octomap(environment, new_pose, obs_levels, obs_sizes,
                                        map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
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
        with timing_measurer.measurement("insert_depth_image"):
            result = environment.get_mapper().perform_insert_depth_map_rpy(
                new_pose.location(), new_pose.orientation_rpy(),
                depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)
        # print("result diff:", sim_result.probabilistic_reward - result.probabilistic_reward)
        # assert(sim_result.probabilistic_reward - result.probabilistic_reward == 0)

        # Query octomap
        with timing_measurer.measurement("query_octomap"):
            out_grid_3ds = query_octomap(environment, new_pose, obs_levels, obs_sizes,
                                         map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)

        print("Selected action={}, probabilistic reward={}".format(action, result.probabilistic_reward))
        print("Grid differences:", [np.sum(out_grid_3ds[..., i] - in_grid_3ds[..., i]) for i in xrange(in_grid_3ds.shape[-1])])

        # Keep record for saving later
        rewards = np.asarray([result.reward, result.normalized_reward,
                            result.probabilistic_reward, result.normalized_probabilistic_reward], dtype=np.float32)
        # scores = np.array([result.score, result.normalized_score,
        #                    result.probabilistic_score, result.normalized_probabilistic_score])
        record = data_record.RecordV4(intrinsics, map_resolution, axis_mode, forward_factor,
                                      obs_levels, in_grid_3ds, out_grid_3ds,
                                      rewards, scores,
                                      rgb_image, depth_image, normal_image)

        prev_action = action

        records.append(record)

        timing_measurer.print_times()

        if len(records) % records_per_file == 0:
            # filename, next_file_num = get_next_output_tf_filename(next_file_num)
            filename, next_file_num = file_helpers.get_next_output_hdf5_filename(
                next_file_num, template=filename_template)
            print("Writing records to file {}".format(filename))
            # write_tf_records(filename, records)
            if not args.dry_run:
                data_record.write_hdf5_records_v4(filename, records, dataset_kwargs=dataset_kwargs)
                if check_written_records:
                    print("Reading records from file {}".format(filename))
                    records_read = data_record.read_hdf5_records_v4_as_list(filename)
                    for record, record_read in zip(records, records_read):
                        assert(np.all(record.intrinsics == record_read.intrinsics))
                        assert(record.map_resolution == record_read.map_resolution)
                        assert(record.axis_mode == record_read.axis_mode)
                        assert(record.forward_factor == record_read.forward_factor)
                        assert(np.all(record.obs_levels == record_read.obs_levels))
                        for in_grid_3d, in_grid_3d_read in zip(record.in_grid_3d, record_read.in_grid_3d):
                            assert(np.all(in_grid_3d == in_grid_3d_read))
                        for out_grid_3d, out_grid_3d_read in zip(record.out_grid_3d, record_read.out_grid_3d):
                            assert(np.all(out_grid_3d == out_grid_3d_read))
                        assert(np.all(record.rewards == record_read.rewards))
                        assert(np.all(record.scores == record_read.scores))
                        assert(np.all(record.rgb_image == record_read.rgb_image))
                        assert(np.all(record.normal_image == record_read.normal_image))
                        assert(np.all(record.depth_image == record_read.depth_image))
            records = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help="Do not save anything")
    parser.add_argument('--output-path', required=True, type=str, help="Output path")
    parser.add_argument('--obs-levels', default="0,1,2,3,4", type=str)
    parser.add_argument('--obs-sizes', default="16,16,16", type=str)
    parser.add_argument('--records-per-file', default=1000, type=int, help="Samples per file")
    parser.add_argument('--num-files', default=100, type=int, help="Number of files")
    parser.add_argument('--environment-config', type=str, required=True, help="Environment configuration file")
    parser.add_argument('--reset-interval', default=100, type=int)
    parser.add_argument('--reset-score-threshold', default=0.3, type=float)
    parser.add_argument('--epsilon', default=0.2, type=float)
    parser.add_argument('--client-id', default=0, type=int)
    parser.add_argument('--wait-until-pose-set', type=argparse_bool, default=True,
                        help="Wait until pose is set in Unreal Engine")
    parser.add_argument('--measure-timing', type=argparse_bool, default=False, help="Measure timing of steps")
    parser.add_argument('--visualize', type=argparse_bool, default=False)
    parser.add_argument('--compression-level', default=5, type=int, help="Gzip compression level")
    parser.add_argument('--check-written-records', type=argparse_bool, default=True,
                        help="Whether written files should be read and checked afterwards.")

    args = parser.parse_args()

    run(args)
