#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
import exceptions
import time
import threading
import Queue
import data_record
import file_helpers
import models
import numpy as np
import yaml
import tensorflow as tf
import tensorflow.contrib.staging as tf_staging
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.memory_stats as tf_memory_stats
import tf_utils
import data_provider
import input_pipeline
import configuration
import visualization
import env_factory
import collect_data
from PIL._imaging import draw
from utils import argparse_bool
from attribute_dict import AttributeDict
from policy_helpers import VisitedPoses
from tensorflow.python.client import timeline
from RLrecon.utils import Timer
from RLrecon import math_utils


class DynamicPlot(object):

    def __init__(self, rows, fig_num=None, marker='-'):
        import matplotlib.pyplot as plt
        self._fig, self._axes = plt.subplots(nrows=rows, num=fig_num)
        if rows == 1:
            self._axes = [self._axes]
        self._plots = []
        for ax in self._axes:
            plot, = ax.plot([], [], marker)
            ax.set_autoscalex_on(True)
            ax.set_autoscaley_on(True)
            ax.grid()
            self._plots.append(plot)
        self._counts = [0 for _ in self._axes]

    def get_axes(self, ax_idx=None):
        if ax_idx is None:
            return self._axes
        else:
            return self._axes[ax_idx]

    def update(self):
        for ax in self._axes:
            ax.relim()
            ax.autoscale_view()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def set_data(self, x_data, y_data, ax_idx=0, update=True):
        plot = self._plots[ax_idx]
        plot.set_xdata(x_data)
        plot.set_ydata(y_data)
        if update:
            self.update()

    def add_point(self, x, y, ax_idx=0, update=True):
        plot = self._plots[ax_idx]
        self.set_data(
            np.concatenate([plot.get_xdata(), [x]]),
            np.concatenate([plot.get_ydata(), [y]]),
            ax_idx, update)

    def add_points(self, x, y, update=True):
        assert (len(x) == len(y))
        assert (len(y) == len(self._axes))
        for i in xrange(len(self._axes)):
            self.add_point(x[i], y[i], ax_idx=i, update=False)
        if update:
            self.update()

    def add_point_auto_x(self, y, ax_idx=0, update=True):
        count = self._counts[ax_idx]
        self._counts[ax_idx] += 1
        self.add_point(count, y, ax_idx, update)

    def add_points_auto_x(self, y, update=True):
        assert (len(y) == len(self._axes))
        for i in xrange(len(self._axes)):
            self.add_point_auto_x(y[i], ax_idx=i, update=False)
        if update:
            self.update()

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        return plt.show(*args, **kwargs)

    def reset(self, update=True):
        for i in xrange(len(self._axes)):
            self.set_data([], [], ax_idx=i, update=False)
        if update:
            self.update()


def predict_reward_with_raycast(environment, pose, max_range, ignore_unknown_voxels=False):
    width = environment._engine.get_width()
    height = environment._engine.get_height()
    focal_length = environment._engine.get_focal_length()
    rr = environment._mapper.perform_raycast_camera_rpy(
        pose.location(),
        pose.orientation_rpy(),
        width, height, focal_length,
        ignore_unknown_voxels,
        max_range
    )
    # return rr.num_hits_unknown
    return rr.expected_reward


def compute_reward(environment, pose, intrinsics=None, downsample_to_grid=True):
    if intrinsics is None:
        intrinsics = environment.get_engine().get_intrinsics()
    environment.set_pose(pose, wait_until_set=True)
    depth_image = environment.get_engine().get_depth_image()
    result = environment.get_mapper().perform_insert_depth_map_rpy(
        pose.location(), pose.orientation_rpy(),
        depth_image, intrinsics, downsample_to_grid=True, simulate=True)
    prob_reward = result.probabilistic_reward
    assert(prob_reward >= 0)
    return prob_reward


def run_episode(environment, predict_reward,
                score_threshold=np.finfo(np.float32).max,
                verbose=False, plot=None, visualize_images=False):
    intrinsics = environment.get_engine().get_intrinsics()

    if visualize_images:
        import cv2

    visited_poses = VisitedPoses(3 + 4, np.concatenate([np.ones((3,)), 10. * np.ones((4,))]))

    print("Starting episode")
    environment.reset()

    i = 0
    while True:
        result = environment.get_mapper().perform_info()
        score = result.normalized_probabilistic_score
        if score >= score_threshold:
            print("Episode finished with score: {}".format(score))
            return

        current_pose = environment.get_pose()
        # environment.set_pose(current_pose, wait_until_set=True)
        visited_poses.add_visited_pose(current_pose)

        print("Step # {}".format(i))

        predicted_rewards = np.zeros((environment.get_num_of_actions(),))
        visit_counts = np.zeros((environment.get_num_of_actions(),))
        collision_flags = np.zeros((environment.get_num_of_actions(),))
        for action in xrange(environment.get_num_of_actions()):
            if verbose:
                print("Evaluating action {}".format(action))
            if environment.is_action_colliding(current_pose, action, verbose=True):
                print("Action {} would collide".format(action))
                collision_flags[action] = 1
            new_pose = environment.simulate_action_on_pose(current_pose, action)
            predicted_reward = predict_reward(new_pose)
            predicted_rewards[action] = predicted_reward
            visit_count = visited_poses.get_visit_count(new_pose)
            visit_counts[action] = visit_count

        # if visualize:
        #     fig = 1
        #     fig = visualization.plot_grid(input[..., 2], input[..., 3], title_prefix="input", show=False, fig_offset=fig)
        #     # fig = visualization.plot_grid(record.in_grid_3d[..., 6], record.in_grid_3d[..., 7], title_prefix="in_grid_3d", show=False, fig_offset=fig)
        #     visualization.show(stop=True)

        print("Predicted rewards:", predicted_rewards)
        visit_counts = np.array(visit_counts, dtype=np.float32)
        visit_weights = 1. / (visit_counts + 1)
        if verbose:
            print("Visit weights: {}".format(visit_weights))
        adjusted_rewards = predicted_rewards * visit_weights
        adjusted_rewards[collision_flags > 0] = - np.finfo(np.float32).max
        print("Adjusted expected rewards:", adjusted_rewards)

        if np.all(collision_flags > 0):
            print("There is no action that does not lead to a collision. Stopping.")
            break

        best_action = None
        if args.interactive:
            valid_input = False
            while not valid_input:
                user_input = raw_input("Overwrite action? [{}-{}] ".format(0, environment.get_num_of_actions()))
                if len(user_input) > 0:
                    try:
                        best_action = int(user_input)
                        if best_action < 0 or best_action >= environment.get_num_of_actions() \
                                or collision_flags[best_action] > 0:
                            print("Invalid action: {}".format(best_action))
                            best_action = None
                        else:
                            valid_input = True
                    except exceptions.ValueError:
                        pass

        if best_action is None:
            # Select best action and perform it
            # if i == 0:
            #     best_action = 0
            # else:
            #     best_action = np.argmax(adjusted_rewards)
            best_action = np.argmax(adjusted_rewards)

        if collision_flags[best_action] > 0:
            print("There is no action that does not lead to a collision. Stopping.")
            break

        if verbose:
            print("Print performing action")
        # best_action = 0
        new_pose = environment.simulate_action_on_pose(current_pose, best_action)
        environment.set_pose(new_pose, wait_until_set=True)
        depth_image = environment.get_engine().get_depth_image()

        # sim_result = environment.get_mapper().perform_insert_depth_map_rpy(
        #     new_pose.location(), new_pose.orientation_rpy(),
        #     depth_image, intrinsics, downsample_to_grid=True, simulate=True)
        result = environment.get_mapper().perform_insert_depth_map_rpy(
            new_pose.location(), new_pose.orientation_rpy(),
            depth_image, intrinsics, downsample_to_grid=True, simulate=False)
        # print("result diff:", sim_result.probabilistic_reward - result.probabilistic_reward)
        # assert(sim_result.probabilistic_reward - result.probabilistic_reward == 0)

        # Output expected and received reward
        score = result.normalized_probabilistic_score
        reward = result.probabilistic_reward
        error = predicted_rewards[best_action] - reward
        print("Selected action={}, expected reward={}, probabilistic reward={}, error={}, score={}".format(
            best_action, predicted_rewards[best_action], reward, error, score))
        if plot is not None:
            if verbose:
                print("Updating plot")
            plot.add_points_auto_x([score, reward, error, error / np.maximum(reward, 1)])

        if visualize_images:
            print("Showing RGB, normal and depth images")

            def draw_selected_pixels(img, pixels):
                radius = 2
                color = (255, 0, 0)
                for pixel in pixels:
                    x = int(np.round(pixel.x))
                    y = int(np.round(pixel.y))
                    cv2.circle(img, (x, y), radius, color)

            rgb_image = environment.get_engine().get_rgb_image()
            normal_rgb_image = environment.get_engine().get_normal_rgb_image()
            rgb_image_show = np.array(rgb_image)
            normal_rgb_image_show = np.array(normal_rgb_image)
            depth_image_show = np.array(depth_image) / 20.
            depth_image_show[depth_image_show > 1] = 1
            if verbose >= 2:
                print("Marking used depth pixels")
                draw_selected_pixels(rgb_image_show, result.used_pixel_coordinates)
                draw_selected_pixels(normal_rgb_image_show, result.used_pixel_coordinates)
                draw_selected_pixels(depth_image_show, result.used_pixel_coordinates)
            cv2.imshow("RGB", rgb_image_show)
            cv2.imshow("Normal", normal_rgb_image_show)
            cv2.imshow("Depth", depth_image_show)
            cv2.waitKey(500 if i == 0 else 100)

        i += 1
        # time.sleep(1)


def run(args):
    # Read config file
    topic_cmdline_mappings = {"tensorflow": "tf"}
    topics = ["tensorflow", "io", "training", "data"]
    cfg = configuration.get_config_from_cmdline(
        args, topics, topic_cmdline_mappings)
    if args.config is not None:
        with file(args.config, "r") as config_file:
            tmp_cfg = yaml.load(config_file)
            configuration.update_config_from_other(cfg, tmp_cfg)

    # Read model config
    if "model" in cfg:
        model_config = cfg["model"]
    elif args.model_config is not None:
        model_config = {}
    else:
        print("ERROR: Model configuration must be in general config file or provided in extra config file.")
        import sys
        sys.exit(1)

    if args.model_config is not None:
        with file(args.model_config, "r") as config_file:
            tmp_model_config = yaml.load(config_file)
            configuration.update_config_from_other(model_config, tmp_model_config)

    cfg = AttributeDict.convert_deep(cfg)
    model_config = AttributeDict.convert_deep(model_config)

    filename_generator = file_helpers.input_filename_generator_hdf5(args.data_path)
    filename = next(filename_generator)
    print("Filename: {}".format(filename))

    # if not cfg.data.input_stats_filename:
    #     cfg.data.input_stats_filename = os.path.join(args.data_path, file_helpers.DEFAULT_HDF5_STATS_FILENAME)
    input_shape, get_input_from_record, target_shape, get_target_from_record = \
        input_pipeline.get_input_and_target_from_record_functions(cfg.data, [filename], args.verbose)

    if args.verbose:
        input_pipeline.print_data_stats(filename, get_input_from_record, get_target_from_record)

    # Configure tensorflow
    tf_config = tf.ConfigProto()
    # tf_config.gpu_options.per_process_gpu_memory_fraction = cfg.tensorflow.gpu_memory_fraction
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
    tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
    tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
    tf_config.log_device_placement = cfg.tensorflow.log_device_placement

    batch_size = 1

    sess = tf.Session(config=tf_config)

    def parse_record(record):
        single_input = get_input_from_record(record)
        single_target = get_target_from_record(record)
        return single_input, single_target

    # Create model
    if len(tf_utils.get_available_gpu_names()) > 0:
        device_name = tf_utils.get_available_gpu_names()[0]
    else:
        device_name = tf_utils.cpu_device_name()
    with tf.device(device_name):
        input_placeholder = tf.placeholder(tf.float32, shape=[batch_size] + list(input_shape), name="in_input")
        target_placeholder = tf.placeholder(tf.float32, shape=[batch_size] + list(target_shape), name="in_target")
        with tf.variable_scope("model"):
            model = models.Model(model_config,
                                 input_placeholder,
                                 target_placeholder,
                                 is_training=False,
                                 verbose=args.verbose)

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(model.global_variables, save_relative_paths=True)

    # Restore model
    if args.checkpoint is None:
        print("Reading latest checkpoint from {}".format(args.model_path))
        ckpt = tf.train.get_checkpoint_state(args.model_path)
        if ckpt is None:
            raise exceptions.IOError("No previous checkpoint found at {}".format(args.model_path))
        print('Found previous checkpoint... restoring')
        checkpoint_path = ckpt.model_checkpoint_path
    else:
        checkpoint_path = os.path.join(args.model_path, args.checkpoint)
    print("Trying to restore model from checkpoint {}".format(checkpoint_path))
    saver.restore(sess, checkpoint_path)

    sess.graph.finalize()

    environment_class = env_factory.get_environment_class_by_name(args.environment)
    client_id = args.client_id
    environment = env_factory.create_environment(environment_class, client_id)

    intrinsics = environment.get_engine().get_intrinsics()
    result = environment.get_mapper().perform_info()
    map_resolution = result.resolution

    axis_mode = 0
    forward_factor = 3 / 8.
    # axis_mode = 1
    # forward_factor = 0.

    obs_levels = [0, 1, 2, 3]
    obs_sizes_x = [16] * len(obs_levels)
    obs_sizes_y = obs_sizes_x
    obs_sizes_z = obs_sizes_x
    obs_sizes = [obs_sizes_x, obs_sizes_y, obs_sizes_z]

    def get_record_from_octomap(pose):
        # Query octomap
        in_grid_3ds = collect_data.query_octomap(
            environment, pose, obs_levels, obs_sizes,
            map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
        record = data_record.RecordV3(obs_levels, in_grid_3ds, None, None, None)
        return record

    def predict_reward(pose):
        record = get_record_from_octomap(pose)
        input = get_input_from_record(record)
        input_batch = input[np.newaxis, ...]
        output_batch, = sess.run(
            [model.output],
            feed_dict={
                input_placeholder: input_batch,
            })
        predicted_reward = output_batch[0, ...]
        return predicted_reward

    # # Raycast heuristic
    # def predict_reward(pose):
    #     max_range = 15
    #     return predict_reward_with_raycast(environment, pose, max_range)

    def predict_true_reward(pose):
        return compute_reward(environment, pose, intrinsics, downsample_to_grid=True)

    if args.visualize:
        plot = DynamicPlot(rows=4)
        plot.get_axes(0).set_ylabel("Score")
        plot.get_axes(1).set_ylabel("Reward")
        plot.get_axes(2).set_ylabel("Error")
        plot.get_axes(3).set_ylabel("Relative error")
        plot.show(block=False)
    else:
        plot = None

    if args.use_oracle:
        predict_reward_fn = predict_true_reward
    else:
        predict_reward_fn = predict_reward

    while True:
        plot.reset()
        run_episode(environment, predict_reward_fn, args.score_threshold,
                    verbose=args.verbose, plot=plot,
                    visualize_images=args.visualize)


if __name__ == '__main__':
    # np.set_printoptions(threshold=5)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--data-path', required=True, help='Data path.')
    parser.add_argument('--model-path', required=True, help='Model path.')
    parser.add_argument('--checkpoint', help='Checkpoint to restore.')
    parser.add_argument('--config', type=str, help='YAML configuration.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')
    parser.add_argument('--visualize', type=argparse_bool, default=False)
    parser.add_argument('--interactive', type=argparse_bool, default=False)
    parser.add_argument('--use-oracle', type=argparse_bool, default=False)
    parser.add_argument('--score-threshold', type=float, default=0.3)

    parser.add_argument('--environment', type=str, required=True, help="Environment name")
    parser.add_argument('--client-id', default=0, type=int)

    # IO
    parser.add_argument('--io.timeout', type=int, default=10 * 60)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=1)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")

    # Data parameters
    parser.add_argument('--data.input_id', type=str, default="in_grid_3d")
    parser.add_argument('--data.target_id', type=str, default="prob_reward")
    parser.add_argument('--data.input_stats_filename', type=str)
    parser.add_argument('--data.obs_levels_to_use', type=str)
    parser.add_argument('--data.subvolume_slice_x', type=str)
    parser.add_argument('--data.subvolume_slice_y', type=str)
    parser.add_argument('--data.subvolume_slice_z', type=str)
    parser.add_argument('--data.normalize_input', type=argparse_bool, default=True)
    parser.add_argument('--data.normalize_target', type=argparse_bool, default=False)
    parser.add_argument('--fake_constant_data', type=argparse_bool, default=False,
                        help='Use constant fake data.')
    parser.add_argument('--fake_random_data', type=argparse_bool, default=False,
                        help='Use constant fake random data.')

    args = parser.parse_args()

    run(args)
