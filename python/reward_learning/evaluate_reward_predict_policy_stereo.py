#!/usr/bin/env python
from __future__ import print_function

import os
import argparse
from builtins import range
from pybh import hdf5_utils
import models
import numpy as np
import yaml
import cv2
import tensorflow as tf
from pybh import file_utils
from pybh import tf_utils
import input_pipeline
import configuration
import env_factory
import collect_data
from pybh.attribute_dict import AttributeDict
from policy_helpers import VisitedPoses
from pybh.utils import Timer, TimeMeter, DummyTimeMeter, argparse_bool, time_measurement
from pybh import math_utils
from pybh import log_utils


logger = log_utils.get_logger("evaluate_policy")


class DynamicPlot(object):

    def __init__(self, rows, fig_num=None, marker='-'):
        import matplotlib
        # matplotlib.use('Qt4Agg')
        # matplotlib.use('GTK3Agg')
        # matplotlib.use('WXAgg')
        matplotlib.use('GTKAgg')
        # matplotlib.use('GTK3Cairo')
        import matplotlib.pyplot as plt
        # plt.style.use('ggplot')
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
            with time_measurement("relim()"):
                ax.relim()
            with time_measurement("autoscale_view()"):
                ax.autoscale_view()
        with time_measurement("canvas_draw()"):
            self._fig.canvas.draw()
            # self._fig.canvas.update()
        with time_measurement("canvas_flush()"):
            self._fig.canvas.flush_events()

    def set_data(self, x_data, y_data, ax_idx=0, update=True):
        plot = self._plots[ax_idx]
        with time_measurement("set_xdata()"):
            plot.set_xdata(x_data)
        with time_measurement("set_ydata()"):
            plot.set_ydata(y_data)
        if update:
            with time_measurement("update()"):
                self.update()

    def add_point(self, x, y, ax_idx=0, update=True):
        plot = self._plots[ax_idx]
        with time_measurement("concatenate x()"):
            new_xdata = np.concatenate([plot.get_xdata(), [x]])
        with time_measurement("concatenate y()"):
            new_ydata = np.concatenate([plot.get_ydata(), [y]])
        with time_measurement("set_data()"):
            self.set_data(new_xdata, new_ydata, ax_idx, update)

    def add_points(self, x, y, update=True):
        assert (len(x) == len(y))
        assert (len(y) == len(self._axes))
        for i in range(len(self._axes)):
            self.add_point(x[i], y[i], ax_idx=i, update=False)
        if update:
            self.update()

    def add_point_auto_x(self, y, ax_idx=0, update=True):
        count = self._counts[ax_idx]
        self._counts[ax_idx] += 1
        self.add_point(count, y, ax_idx, update)

    def add_points_auto_x(self, y, update=True):
        assert (len(y) == len(self._axes))
        with time_measurement("add_point_auto_x()"):
            for i in range(len(self._axes)):
                self.add_point_auto_x(y[i], ax_idx=i, update=False)
        if update:
            with time_measurement("update()"):
                self.update()

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        return plt.show(*args, **kwargs)

    def reset(self, update=True):
        for i in range(len(self._axes)):
            self.set_data([], [], ax_idx=i, update=False)
        self._counts = [0 for _ in self._axes]
        if update:
            self.update()


def predict_reward_with_raycast(environment, pose, max_range, ignore_unknown_voxels=False, timer=None):
    if timer is None:
        timer = DummyTimeMeter()
    with timer.measure("get_engine_intrinsics"):
        width = environment.base.engine.get_width()
        height = environment.base.engine.get_height()
        focal_length = environment.base.engine.get_focal_length()
    with timer.measure("perform_raycast"):
        rr = environment.base.mapper.perform_raycast_camera_rpy(
            pose.location(),
            pose.orientation_rpy(),
            width, height, focal_length,
            ignore_unknown_voxels,
            max_range
        )
    # return rr.num_hits_unknown
    predicted_depth_image = None
    return rr.expected_reward, predicted_depth_image


def compute_true_reward(environment, pose, intrinsics=None, downsample_to_grid=True, timer=None):
    if timer is None:
        timer = DummyTimeMeter()
    if intrinsics is None:
        with timer.measure("get_intrinsics"):
            intrinsics = environment.get_engine().get_intrinsics()
    with timer.measure("set_pose"):
        environment.base.set_pose(pose, wait_until_set=True)
    # environment.base.get_engine().get_depth_image()
    with timer.measure("get_depth_image"):
        depth_image = environment.base.get_engine().get_depth_image()
    with timer.measure("insert_depth_image"):
        result = environment.base.get_mapper().perform_insert_depth_map_rpy(
            pose.location(), pose.orientation_rpy(),
            depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=True)
    prob_reward = result.probabilistic_reward
    assert prob_reward >= 0
    return prob_reward, depth_image


def compute_predicted_action_rewards(environment, visited_poses, pose, plan_depth, compute_reward_fn,
                                     tmp_rewards_list, tmp_actions_list, depth_images,
                                     local_future_rewards, is_oracle,
                                     intrinsics, downsample_to_grid,
                                     print_prefix="", keep_depth=False,
                                     time_meter=DummyTimeMeter(),
                                     verbose=False,
                                     do_insert_depth_map_simulate_sanity_check=False,
                                     do_2step_sanity_check=False):
    predicted_rewards = np.zeros((environment.action_space.n,))
    adjusted_rewards = np.zeros((environment.action_space.n,))
    visit_counts = np.zeros((environment.action_space.n,))
    collision_flags = np.zeros((environment.action_space.n,))

    for action in range(environment.action_space.n):

        if verbose:
            logger.info(print_prefix + "Evaluating action {} ({})".format(action, plan_depth))
        with time_meter.measure("simulate_action_on_pose"):
            new_pose = environment.base.simulate_action_on_pose(pose, action)
        logger.info("new pose within: {}".format(new_pose))
        predicted_reward, predicted_depth_image = compute_reward_fn(new_pose, timer=time_meter)
        if do_insert_depth_map_simulate_sanity_check:
            predicted_reward2, predicted_depth_image2 = compute_reward_fn(new_pose)
            if predicted_reward != predicted_reward2:
                for i in range(25):
                    predicted_reward3, predicted_depth_image3 = compute_reward_fn(new_pose)
                    if predicted_reward != predicted_reward2:
                        logger.info("predicted_reward3:", predicted_reward3)
                        logger.info("img_diff:", np.sum(np.abs(predicted_depth_image - predicted_depth_image3)))
                logger.info("predicted_reward:", predicted_reward)
                logger.info("predicted_reward2:", predicted_reward2)
                logger.info("img_diff:", np.sum(np.abs(predicted_depth_image - predicted_depth_image2)))
                import cv2
                predicted_depth_image[predicted_depth_image > 20] = 20.
                predicted_depth_image /= 20
                predicted_depth_image2[predicted_depth_image2 > 20] = 20.
                predicted_depth_image2 /= 20
                cv2.imwrite("depth1.png", predicted_depth_image * 255)
                cv2.imwrite("depth2.png", predicted_depth_image2 * 255)
                cv2.imshow("depth1", predicted_depth_image)
                cv2.imshow("depth2", predicted_depth_image2)
                cv2.waitKey()
                assert predicted_reward == predicted_reward2
        if keep_depth and do_insert_depth_map_simulate_sanity_check:
            depth_images[action] = predicted_depth_image
        visit_count = float(visited_poses.get_visit_count(new_pose))
        visit_weight = 1. / (visit_count + 1)
        adjusted_reward = predicted_reward * visit_weight
        if not args.ignore_collision and environment.base.is_action_colliding(pose, action, verbose=True):
            logger.info(print_prefix + "  Action {} would collide".format(action))
            collision_flags[action] = 1
            adjusted_reward = - np.finfo(np.float32).max
        logger.info(
            print_prefix + "Predicted reward for action {} ({}): {}. Collision: {}. Adjusted reward: {}".format(
                action, plan_depth, predicted_reward, collision_flags[action], adjusted_reward))

        if plan_depth > 0 and collision_flags[action] <= 0:
            with time_meter.measure("increase_visit_count"):
                visited_poses.increase_visited_pose(new_pose)
            if predicted_depth_image is not None:
                with time_meter.measure("push_octomap"):
                    environment.base.get_mapper().perform_push_octomap()
                with time_meter.measure("insert_depth_image"):
                    result = environment.base.get_mapper().perform_insert_depth_map_rpy(
                        new_pose.location(), new_pose.orientation_rpy(),
                        predicted_depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)
            else:
                logger.info("WARNING: Doing multistep planning without depth map prediction")
            if is_oracle and do_insert_depth_map_simulate_sanity_check:
                if result.probabilistic_reward != predicted_reward:
                    logger.info(print_prefix + "result.probabilistic_reward:", result.probabilistic_reward)
                    logger.info(print_prefix + "predicted_reward:", predicted_reward)
                assert result.probabilistic_reward == predicted_reward
            logger.info("new pose before: ", new_pose)
            next_step_predicted_rewards, next_step_adjusted_rewards, \
            next_step_visit_counts, next_step_collision_flags = \
                compute_predicted_action_rewards(environment, visited_poses, new_pose, plan_depth - 1,
                                                 compute_reward_fn, tmp_rewards_list, tmp_actions_list, depth_images,
                                                 local_future_rewards, is_oracle, intrinsics, downsample_to_grid,
                                                 print_prefix + "    ", keep_depth, time_meter, verbose,
                                                 do_insert_depth_map_simulate_sanity_check, do_2step_sanity_check)
            logger.info("new pose after: ", new_pose)

            if do_2step_sanity_check:
                next_step_tmp_rewards = np.array(next_step_adjusted_rewards)
                next_step_tmp_rewards[next_step_collision_flags > 0] = -np.inf
                i = np.argmax(next_step_tmp_rewards)
                logger.info(print_prefix + "Next best action for action {} is {}".format(action, i))
                tmp_rewards_list[action] = next_step_predicted_rewards[i]
                tmp_actions_list[action] = i

            adjusted_reward += np.max(next_step_adjusted_rewards)
            if predicted_depth_image is not None:
                with time_meter.measure("pop_octomap"):
                    environment.base.get_mapper().perform_pop_octomap()
            with time_meter.measure("decrease_visit_count"):
                visited_poses.decrease_visited_pose(new_pose)

        predicted_rewards[action] = predicted_reward
        adjusted_rewards[action] = adjusted_reward
        visit_counts[action] = visit_count

    if do_2step_sanity_check:
        if plan_depth > 0:
            tmp_rewards = np.array(adjusted_rewards)
            tmp_rewards[collision_flags > 0] = -np.inf
            i = np.argmax(tmp_rewards)
            local_future_rewards[0] = predicted_rewards[i]

    return predicted_rewards, adjusted_rewards, visit_counts, collision_flags


def compute_predicted_action_rewards2(environment, visited_poses, pose, plan_depth, compute_reward_fn,
                                     tmp_rewards_list, tmp_actions_list, depth_images,
                                     local_future_rewards, is_oracle,
                                     intrinsics, downsample_to_grid,
                                     print_prefix="", keep_depth=False,
                                     time_meter=DummyTimeMeter(),
                                     verbose=False,
                                     do_insert_depth_map_simulate_sanity_check=False,
                                     do_2step_sanity_check=False):
    predicted_rewards = np.zeros((environment.action_space.n,))
    adjusted_rewards = np.zeros((environment.action_space.n,))
    visit_counts = np.zeros((environment.action_space.n,))
    collision_flags = np.zeros((environment.action_space.n,))
    all_predicted_reward, _ = compute_reward_fn(pose, timer=time_meter)

    for action in range(environment.action_space.n):

        if verbose:
            logger.info(print_prefix + "Evaluating action {} ({})".format(action, plan_depth))
        with time_meter.measure("simulate_action_on_pose"):
            new_pose = environment.base.simulate_action_on_pose(pose, action)
            logger.info("new pose within: ", new_pose)
        predicted_reward = all_predicted_reward[action]
        visit_count = float(visited_poses.get_visit_count(new_pose))
        visit_weight = 1. / (visit_count + 1)
        adjusted_reward = predicted_reward * visit_weight
        if not args.ignore_collision and environment.base.is_action_colliding(pose, action, verbose=True):
            logger.info(print_prefix + "  Action {} would collide".format(action))
            collision_flags[action] = 1
            adjusted_reward = - np.finfo(np.float32).max
        logger.info(
            print_prefix + "Predicted reward for action {} ({}): {}. Collision: {}. Adjusted reward: {}".format(
                action, plan_depth, predicted_reward, collision_flags[action], adjusted_reward))

        predicted_rewards[action] = predicted_reward
        adjusted_rewards[action] = adjusted_reward
        visit_counts[action] = visit_count

    return predicted_rewards, adjusted_rewards, visit_counts, collision_flags


def run_episode(args, episode, environment, compute_reward_fn, policy_mode,
                reset_interval=np.iinfo(np.int32).max,
                reset_score_threshold=np.finfo(np.float32).max,
                downsample_to_grid=True,
                is_oracle=False,
                verbose=False, plot=None, visualize_images=False,
                measurement_hook=None,
                measure_timing=False,
                **kwargs):
    if measure_timing:
        timer = Timer()
        time_meter = TimeMeter()
    else:
        time_meter = DummyTimeMeter()

    intrinsics = environment.base.get_engine().get_intrinsics()

    if visualize_images:
        import cv2

    visited_poses = VisitedPoses(3 + 4, np.concatenate([np.ones((3,)), 10. * np.ones((4,))]))

    logger.info("Starting episode")
    environment.reset()

    output = {
        "score": [],
        "computed_reward": [],
        "true_reward": [],
        "pose": [],
    }

    do_insert_depth_map_simulate_sanity_check = False
    do_2step_sanity_check = False

    if args.plan_steps == 1:
        do_2step_sanity_check = False

    if do_2step_sanity_check:
        assert args.plan_steps == 2
        future_rewards = [0]
        future_actions = [0]

    i = 0
    while True:
        logger.info("Getting info")
        result = environment.base.get_mapper().perform_info()
        logger.info("Done")
        score = result.normalized_probabilistic_score
        if score >= reset_score_threshold or i >= reset_interval:
            logger.info("Episode finished with score {} after {} steps".format(score, i))
            print("Episode took {} seconds".format(timer.elapsed_seconds()))
            return output

        current_pose = environment.base.get_pose()
        # environment.base.set_pose(current_pose, wait_until_set=True)
        visited_poses.increase_visited_pose(current_pose)

        if i == 0:
            # Initialize output
            output["score"].append(score)
            output["computed_reward"].append(0)
            output["true_reward"].append(0)
            output["pose"].append(current_pose)
            logger.info("Initial pose: {}".format(current_pose))

        logger.info("Step # {} of episode # {} with policy {} and plan depth {}".format(i, episode, args.policy_mode, args.plan_steps))

        if do_insert_depth_map_simulate_sanity_check:
            depth_images = [None for _ in range(environment.action_space.n)]
        else:
            depth_images = None

        if do_2step_sanity_check:
            logger.info("WARNING: Doing 2step sanity checks. This causes a lot of overhead.")
            tmp_rewards_list = [-np.inf for _ in range(environment.action_space.n)]
            tmp_actions_list = [0 for _ in range(environment.action_space.n)]
            local_future_rewards = [np.nan]
        else:
            tmp_rewards_list = None
            tmp_actions_list = None
            local_future_rewards = None

        print_prefix = ""
        keep_depth = True
        if do_2step_sanity_check:
            predicted_rewards_1step, adjusted_rewards_1step, visit_counts_1step, collision_flags_1step = \
                compute_predicted_action_rewards(environment, visited_poses, current_pose, args.plan_steps - 2,
                                                 compute_reward_fn, tmp_rewards_list, tmp_actions_list, depth_images,
                                                 local_future_rewards, is_oracle, intrinsics, downsample_to_grid,
                                                 print_prefix, keep_depth, time_meter, verbose,
                                                 do_insert_depth_map_simulate_sanity_check, do_2step_sanity_check)
            depth_images_1step = depth_images
            best_action_1step = np.argmax(adjusted_rewards_1step)
            reward_1step = predicted_rewards_1step[best_action_1step]

        if policy_mode == "action_prediction":
            predicted_rewards, adjusted_rewards, visit_counts, collision_flags = \
                compute_predicted_action_rewards2(environment, visited_poses, current_pose, args.plan_steps - 1,
                                                  compute_reward_fn, tmp_rewards_list, tmp_actions_list, depth_images,
                                                  local_future_rewards, is_oracle, intrinsics, downsample_to_grid,
                                                  print_prefix, keep_depth, time_meter, verbose,
                                                  do_insert_depth_map_simulate_sanity_check, do_2step_sanity_check)
        else:
            predicted_rewards, adjusted_rewards, visit_counts, collision_flags = \
                compute_predicted_action_rewards(environment, visited_poses, current_pose, args.plan_steps - 1,
                                                 compute_reward_fn, tmp_rewards_list, tmp_actions_list, depth_images,
                                                 local_future_rewards, is_oracle, intrinsics, downsample_to_grid,
                                                 print_prefix, keep_depth, time_meter, verbose,
                                                 do_insert_depth_map_simulate_sanity_check, do_2step_sanity_check)

        if do_2step_sanity_check and do_insert_depth_map_simulate_sanity_check:
            for action in range(environment.action_space.n):
                if predicted_rewards[action] != predicted_rewards_1step[action]:
                    logger.info("action:", action)
                    logger.info("predicted_rewards[action]:", predicted_rewards[action])
                    logger.info("predicted_rewards_1step[action]:", predicted_rewards_1step[action])
                    new_pose = environment.base.simulate_action_on_pose(current_pose, action)
                    predicted_reward, predicted_depth_image = compute_reward_fn(new_pose)
                    logger.info("predicted_reward:", predicted_reward)
                    new_pose = environment.base.simulate_action_on_pose(current_pose, action)
                    predicted_reward, predicted_depth_image = compute_reward_fn(new_pose)
                    logger.info("predicted_reward2:", predicted_reward)
                    logger.info("np.sum(np.abs(depth_image - depth_images_1step[best_action])):", np.sum(np.abs(predicted_depth_image - depth_images_1step[action])))
                    logger.info("np.sum(np.abs(depth_image - depth_images[best_action])):", np.sum(np.abs(predicted_depth_image - depth_images[action])))
                    assert False

        # predicted_rewards = np.zeros((environment.action_space.n,))
        # visit_counts = np.zeros((environment.action_space.n,))
        # collision_flags = np.zeros((environment.action_space.n,))
        # for action in range(environment.action_space.n):
        #     if verbose:
        #         logger.info(print_prefix + "Evaluating action {}".format(action))
        #     if environment.base.is_action_colliding(current_pose, action, verbose=True):
        #         logger.info(print_prefix + "Action {} would collide".format(action))
        #         collision_flags[action] = 1
        #     new_pose = environment.base.simulate_action_on_pose(current_pose, action)
        #     predicted_reward, _ = compute_reward_fn(new_pose)
        #     predicted_rewards[action] = predicted_reward
        #     visit_count = visited_poses.get_visit_count(new_pose)
        #     visit_counts[action] = visit_count

        # if visualize:
        #     fig = 1
        #     fig = visualization.plot_grid(input[..., 2], input[..., 3], title_prefix="input", show=False, fig_offset=fig)
        #     # fig = visualization.plot_grid(record.in_grid_3d[..., 6], record.in_grid_3d[..., 7], title_prefix="in_grid_3d", show=False, fig_offset=fig)
        #     visualization.show(stop=True)

        logger.info("Predicted rewards: {}".format(predicted_rewards))
        # visit_counts = np.array(visit_counts, dtype=np.float32)
        if verbose:
            visit_weights = 1. / (visit_counts + 1)
            logger.info("Visit weights: {}".format(visit_weights))
        # adjusted_rewards = predicted_rewards * visit_weights
        # adjusted_rewards[collision_flags > 0] = - np.finfo(np.float32).max
        logger.info("Adjusted expected rewards: {}".format(adjusted_rewards))

        if np.all(collision_flags > 0):
            logger.info("There is no action that does not lead to a collision. Stopping.")
            break

        perform_measurement = True
        best_action = None
        if args.interactive:
            valid_action_ids = np.arange(collision_flags.shape[0])[collision_flags <= 0]
            logger.info("Valid actions: {}".format(valid_action_ids))
            valid_input = False
            while not valid_input:
                user_input = raw_input("Overwrite action? [{}-{}]. Or take measurement? [m] Or ignore? [i] ".format(0, environment.action_space.n))
                if len(user_input) > 0:
                    if user_input == "m":
                        best_action = -1
                        perform_measurement = True
                        valid_input = True
                    elif user_input == "i":
                        valid_input = True
                    elif user_input == "h":
                        logger.info(environment.base.get_pose())
                        best_action = -1
                        logger.info(current_pose)
                        logger.info("Computing reward at current pose.")
                        current_predicted_reward, _ = compute_reward_fn(current_pose, timer=time_meter)
                        logger.info("  reward={}".format(current_predicted_reward))
                    elif user_input == "g":
                        best_action = -1
                        logger.info("Computing reward at current pose.")
                        pose = environment.base.Pose([0, 0, 0], [0, 0, 0])
                        current_predicted_reward, _ = compute_reward_fn(pose, timer=time_meter)
                        logger.info("  reward={}".format(current_predicted_reward))
                    else:
                        try:
                            best_action = int(user_input)
                            if best_action < -1 or best_action >= environment.action_space.n \
                                    or collision_flags[best_action] > 0:
                                logger.info("Invalid action: {}".format(best_action))
                                best_action = None
                            else:
                                valid_input = True
                            perform_measurement = False
                        except ValueError:
                            pass

        # Select best action and perform it

        if best_action is None:
            # best_action = np.argmax(adjusted_rewards)
            max_adjusted_reward = np.max(adjusted_rewards)
            actions = np.arange(environment.action_space.n)[adjusted_rewards == max_adjusted_reward]
            if len(actions) == 1:
                best_action = actions[0]
            else:
                # If there is not a single best action, choose a random one among the best ones.
                best_action = np.random.choice(actions)

        if best_action >= 0:
            assert collision_flags[best_action] <= 0

        if best_action >= 0:
            predicted_reward = predicted_rewards[best_action]
        else:
            predicted_reward = np.nan

        if do_2step_sanity_check:
            future_rewards.append(tmp_rewards_list[best_action])
            future_actions.append(tmp_actions_list[best_action])
            logger.info("Next best action {} will give reward {}".format(tmp_actions_list[best_action], tmp_rewards_list[best_action]))
            logger.info("local_future_rewards:", local_future_rewards)

        if verbose:
            logger.info("Performing action {}".format(best_action))
        # best_action = 0
        with time_meter.measure("simulate_action"):
            new_pose = environment.base.simulate_action_on_pose(current_pose, best_action)
        with time_meter.measure("set_pose"):
            environment.base.set_pose(new_pose, wait_until_set=True)
        with time_meter.measure("get_depth_image"):
            depth_image = environment.base.get_engine().get_depth_image()

        if perform_measurement:
            # sim_result = environment.base.get_mapper().perform_insert_depth_map_rpy(
            #     new_pose.location(), new_pose.orientation_rpy(),
            #     depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=True)
            with time_meter.measure("insert_depth_image"):
                if do_insert_depth_map_simulate_sanity_check:
                    logger.info("WARNING: Doing insert depth map simulation sanity check. This causes a lot of overhead.")
                    simulated_result = environment.base.get_mapper().perform_insert_depth_map_rpy(
                        new_pose.location(), new_pose.orientation_rpy(),
                        depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=True)
                result = environment.base.get_mapper().perform_insert_depth_map_rpy(
                    new_pose.location(), new_pose.orientation_rpy(),
                    depth_image, intrinsics, downsample_to_grid=downsample_to_grid, simulate=False)
                if do_insert_depth_map_simulate_sanity_check:
                    if simulated_result.probabilistic_reward != result.probabilistic_reward:
                        logger.info("simulated_result.probabilistic_reward:", simulated_result.probabilistic_reward)
                        logger.info("result.probabilistic_reward:", result.probabilistic_reward)
                    assert simulated_result.probabilistic_reward == result.probabilistic_reward

            if measurement_hook is not None:
                measurement_hook(new_pose, depth_image, intrinsics)
            # logger.info("result diff:", sim_result.probabilistic_reward - result.probabilistic_reward)
            # assert sim_result.probabilistic_reward - result.probabilistic_reward == 0

            # Output expected and received reward
            score = result.normalized_probabilistic_score
            reward = result.probabilistic_reward
            if do_insert_depth_map_simulate_sanity_check:
                logger.info("np.sum(np.abs(depth_image - depth_images[best_action])):", np.sum(np.abs(depth_image - depth_images[best_action])))
            if is_oracle and do_insert_depth_map_simulate_sanity_check:
                if reward != predicted_reward:
                    logger.info("reward:", reward)
                    logger.info("predicted_reward:", predicted_reward)
                    logger.info("np.sum(np.abs(depth_image - depth_images[best_action])):", np.sum(np.abs(depth_image - depth_images[best_action])))
                    if do_insert_depth_map_simulate_sanity_check:
                        import cv2
                        depth_image[depth_image > 20.] = 20.
                        depth_image /= 20
                        depth_images[best_action][depth_images[best_action] > 20.] = 20.
                        depth_images[best_action] /= 20
                        cv2.imwrite("depth1.png", depth_image * 255)
                        cv2.imwrite("depth2.png", depth_images[best_action] * 255)
                assert reward == predicted_reward
            error = predicted_reward - reward
            logger.info("Selected action={}, expected reward={}, probabilistic reward={}, error={}, score={}".format(
                best_action, predicted_reward, reward, error, score))

            if plot is not None:
                with time_meter.measure("plot_update"):
                    if verbose:
                        logger.info("Updating plot")
                    plot.add_points_auto_x([score, reward, error, error / np.maximum(reward, 1)])

            # Record result for this step
            output["score"].append(score)
            output["computed_reward"].append(predicted_reward)
            output["true_reward"].append(reward)
            output["pose"].append(new_pose)

            # future_rewards.append(local_future_rewards[1])
            # future_adjusted_rewards[best_action].append(local_future_adjusted_rewards[1])
            # future_visit_counts[best_action].append(local_future_visit_counts[1])
            # future_collision_flags[best_action].append(local_future_collision_flags[1])

            if do_2step_sanity_check:
                if local_future_rewards[0] != reward:
                    logger.info("WARNING: Got different reward when performing measurement")
                    logger.info("reward:", reward)
                    logger.info("local_future_rewards[0]:", local_future_rewards[0])
                    if i > 0: assert False
                if future_actions[0] != best_action_1step or future_rewards[0] != reward_1step:
                    logger.info("WARNING: Got different reward/action when performing measurement")
                    logger.info("best_action_1step:", best_action_1step)
                    logger.info("future_actions[0]:", future_actions[0])
                    logger.info("reward_1step:", reward_1step)
                    logger.info("future_rewards[0]:", future_rewards[1])
                    if i > 0: assert False
        if do_2step_sanity_check:
            del future_rewards[0]
            del future_actions[0]

        if visualize_images:
            logger.info("Showing RGB, normal and depth images")

            def draw_selected_pixels(img, pixels):
                radius = 2
                color = (255, 0, 0)
                for pixel in pixels:
                    x = int(np.round(pixel.x))
                    y = int(np.round(pixel.y))
                    cv2.circle(img, (x, y), radius, color)

            rgb_image = environment.base.get_engine().get_rgb_image()
            normal_rgb_image = environment.base.get_engine().get_normal_rgb_image()
            rgb_image_show = np.array(rgb_image)
            normal_rgb_image_show = np.array(normal_rgb_image)
            depth_image_show = np.array(depth_image) / 20.
            depth_image_show[depth_image_show > 1] = 1
            if verbose >= 2:
                logger.info("Marking used depth pixels")
                draw_selected_pixels(rgb_image_show, result.used_pixel_coordinates)
                draw_selected_pixels(normal_rgb_image_show, result.used_pixel_coordinates)
                draw_selected_pixels(depth_image_show, result.used_pixel_coordinates)
            cv2.imshow("RGB", rgb_image_show)
            cv2.imshow("Normal", normal_rgb_image_show)
            cv2.imshow("Depth", depth_image_show)
            cv2.waitKey(500 if i == 0 else 100)

        time_meter.print_times()

        i += 1
        # time.sleep(1)

        # if i == 70:
        #     raw_input("Reached step 70")

    print("Episode took {} seconds".format(timer.elapsed_seconds()))

    return output


def run(args):
    # Create environment
    client_id = args.client_id
    prng_seed = args.prng_seed
    if prng_seed is not None:
        logger.info("Using manual seed for pseudo random number generator: {}".format(prng_seed))
    with open(args.environment_config, "r") as fin:
        environment_config = yaml.load(fin)
    environment = env_factory.create_environment_from_config(environment_config, client_id,
                                                             use_openai_wrapper=True,
                                                             prng_seed=prng_seed)
    stereo_method = environment_config["camera"]["stereo_method"]
    stereo_baseline = environment_config["camera"]["stereo_baseline"]

    intrinsics = environment.base.get_engine().get_intrinsics()
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

    # Load model if in prediction mode
    if args.policy_mode == "prediction" or args.policy_mode == "action_prediction":
        if args.model_path is None:
            raise RuntimeError("Model path is required when using prediction policy mode.")

        # Read config file
        topic_cmdline_mappings = {"tensorflow": "tf"}
        topics = ["tensorflow", "io", "training", "data"]
        cfg = configuration.get_config_from_cmdline(
            args, topics, topic_cmdline_mappings)
        if args.config is not None:
            with open(args.config, "r") as config_file:
                tmp_cfg = yaml.load(config_file)
                configuration.update_config_from_other(cfg, tmp_cfg)

        # Read model config
        if "model" in cfg:
            model_config = cfg["model"]
        elif args.model_config is not None:
            model_config = {}
        else:
            logger.info("ERROR: Model configuration must be in general config file or provided in extra config file.")
            import sys
            sys.exit(1)

        if args.model_config is not None:
            with open(args.model_config, "r") as config_file:
                tmp_model_config = yaml.load(config_file)
                configuration.update_config_from_other(model_config, tmp_model_config)

        cfg = AttributeDict.convert_deep(cfg)

        model_config = AttributeDict.convert_deep(model_config)

        # Retrieve input and target shape
        data_stats_dict = hdf5_utils.read_hdf5_file_to_numpy_dict(args.hdf5_data_stats_path)
        input_and_target_retriever = input_pipeline.InputAndTargetFromData(
            cfg.data, data_stats_dict, batch_data=False, verbose=args.verbose)
        data_sample = {key: array["mean"] for key, array in data_stats_dict.items()}
        input_shape = input_and_target_retriever.get_input_from_data(data_sample).shape
        target_shape = input_and_target_retriever.get_target_from_data(data_sample).shape

        if args.verbose:
            logger.info("Input and target shapes:")
            logger.info("  Shape of input: {}".format(input_shape))
            logger.info("  Shape of target: {}".format(target_shape))

        # Configure tensorflow
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.per_process_gpu_memory_fraction = cfg.tensorflow.gpu_memory_fraction
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        tf_config.intra_op_parallelism_threads = cfg.tensorflow.intra_op_parallelism
        tf_config.inter_op_parallelism_threads = cfg.tensorflow.inter_op_parallelism
        tf_config.log_device_placement = cfg.tensorflow.log_device_placement

        batch_size = 1

        sess = tf.Session(config=tf_config)

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

                if args.policy_mode == "action_prediction":
                    final_conv_module_index = model.module_names.index("conv3d_module_0")
                    final_conv_module = model.modules[final_conv_module_index]
                    with tf.variable_scope("policy"):
                        hidden_layer1 = tf_utils.fully_connected(final_conv_module.output, num_units=512,
                                                                 add_bias=False, use_batch_norm=True, name="hidden_layer1")
                        hidden_layer2 = tf_utils.fully_connected(hidden_layer1, num_units=512,
                                                                 add_bias=False, use_batch_norm=True, name="hidden_layer2")
                        action_logits_model = tf_utils.fully_connected(hidden_layer2, num_units=environment.action_space.n,
                                                                       activation_fn=None, add_bias=True, use_batch_norm=False,
                                                                       name="action_logits")
                        action_probs_model = tf.nn.softmax(action_logits_model, name="action_probs") + 1e-8

        # Restore model
        model_variables = model.global_variables
        if args.policy_mode == "action_prediction":
            model_variables = model_variables + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "model/policy")
        saver = tf.train.Saver(model_variables)
        if args.checkpoint is None:
            logger.info("Reading latest checkpoint from {}".format(args.model_path))
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            if ckpt is None:
                raise IOError("No previous checkpoint found at {}".format(args.model_path))
            logger.info('Found previous checkpoint... restoring')
            checkpoint_path = ckpt.model_checkpoint_path
        else:
            checkpoint_path = os.path.join(args.model_path, args.checkpoint)
        logger.info("Trying to restore model from checkpoint {}".format(checkpoint_path))
        saver.restore(sess, checkpoint_path)

        sess.graph.finalize()

        obs_levels = [int(x.strip()) for x in args.obs_levels.strip("[]").split(",")]
        obs_sizes = [int(x.strip()) for x in args.obs_sizes.strip("[]").split(",")]
        logger.info("obs_levels={}".format(obs_levels))
        logger.info("obs_sizes={}".format(obs_sizes))

        def get_data_from_octomap(pose):
            # Query octomap
            in_grid_3ds = collect_data.query_octomap(
                environment.base, pose, obs_levels, obs_sizes,
                map_resolution, axis_mode=axis_mode, forward_factor=forward_factor)
            data = {"in_grid_3ds": in_grid_3ds[np.newaxis, ...]}
            return data

    # Learned reward
    def compute_predicted_reward(pose, timer=None):
        if timer is None:
            timer = DummyTimeMeter()
        with timer.measure("get_data_from_octomap"):
            data = get_data_from_octomap(pose)
        with timer.measure("get_input_from_data"):
            input_batch = input_and_target_retriever.get_input_from_data(data)
            input_batch = input_and_target_retriever.normalize_input(input_batch)
        # input_batch = input[np.newaxis, ...]
        with timer.measure("evaluate_model"):
            output_batch, = sess.run(
                [model.output],
                feed_dict={
                    input_placeholder: input_batch,
                })
        output = output_batch[0, ...]
        predicted_reward = input_and_target_retriever.denormalize_target(output)
        predicted_depth_image = None
        return predicted_reward, predicted_depth_image

    # Learned reward
    def compute_action_prediction_reward(pose, timer=None):
        if timer is None:
            timer = DummyTimeMeter()
        with timer.measure("get_data_from_octomap"):
            data = get_data_from_octomap(pose)
        with timer.measure("get_input_from_data"):
            input_batch = input_and_target_retriever.get_input_from_data(data)
            input_batch = input_and_target_retriever.normalize_input(input_batch)
        # input_batch = input[np.newaxis, ...]
        with timer.measure("evaluate_model"):
            output_batch, = sess.run(
                [action_probs_model],
                feed_dict={
                    input_placeholder: input_batch,
                })
        output = output_batch[0, ...]
        predicted_reward = output
        predicted_depth_image = None
        return predicted_reward, predicted_depth_image

    # Raycast heuristic
    def compute_heuristic_reward(pose, timer=None):
        return predict_reward_with_raycast(environment, pose, raycast_max_range, timer=timer)

    # True reward
    def compute_oracle_reward(pose, timer=None):
        return compute_true_reward(environment, pose, intrinsics, downsample_to_grid=downsample_to_grid, timer=timer)

    # True reward for blind oracle (i.e. no depth map prediction)
    def compute_blind_oracle_reward(pose, timer=None):
        predicted_reward, _ = compute_oracle_reward(pose, timer)
        predicted_depth_image = None
        return predicted_reward, predicted_depth_image

    # Uniform reward for random exploration (with penalty for already visited poses)
    def compute_uniform_reward(pose, timer=None):
        predicted_depth_image = None
        return 1., predicted_depth_image

    # Zero reward for random exploration (without penalty for already visited poses)
    def compute_zero_reward(pose, timer=None):
        predicted_depth_image = None
        return 0., predicted_depth_image

    # Use RPG IG services to compute reward
    def rpg_ig_function_factory(ig_mode):
        import rospy
        from geometry_msgs.msg import Pose as PoseMsg, Transform as TransformMsg
        from pybh import ros_utils
        from pybh import transforms
        from ig_active_reconstruction_msgs.srv import InformationGainCalculation, InformationGainCalculationRequest
        from ig_active_reconstruction_msgs.srv import DepthMapInput, DepthMapInputRequest
        from ig_active_reconstruction_msgs.srv import ResetOctomap, ResetOctomapRequest
        from ig_active_reconstruction_msgs.srv import SetScoreBoundingBox, SetScoreBoundingBoxRequest
        compute_ig_service = rospy.ServiceProxy("world/information_gain", InformationGainCalculation, persistent=True)
        depth_map_input_service = rospy.ServiceProxy("world/depth_input", DepthMapInput, persistent=True)
        reset_octomap_service = rospy.ServiceProxy("world/reset_octomap", ResetOctomap, persistent=True)
        set_score_bounding_box_service = rospy.ServiceProxy("world/set_score_bounding_box", SetScoreBoundingBox, persistent=True)

        def convert_pose_to_ros_pose_msg(pose):
            pose_msg = PoseMsg()
            ros_utils.point_numpy_to_ros(pose.location(), pose_msg.position)
            orientation_quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
            ros_utils.quaternion_numpy_to_ros(orientation_quat, pose_msg.orientation)
            return pose_msg

        rot_x = transforms.Rotation.x_rotation(np.pi / 2.0)
        rot_z = transforms.Rotation.z_rotation(np.pi / 2.0)

        def rotate_camera_to_rpg_ig_camera(pose):
            quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
            # new_quat = quat
            rot = transforms.Rotation(quat)
            rot = rot.apply_to(rot_z)
            rot = rot.apply_to(rot_x)
            new_orientation_rpy = math_utils.convert_quat_to_rpy(rot.quaternion)
            return environment.base.Pose(pose.location(), new_orientation_rpy)

        def compute_rpg_ig_reward(pose, timer=None):
            if timer is None:
                timer = DummyTimeMeter()
            logger.info("Computing RPG IG")
            request = InformationGainCalculationRequest()
            rpg_ig_pose = rotate_camera_to_rpg_ig_camera(pose)
            pose_msg = convert_pose_to_ros_pose_msg(rpg_ig_pose)
            request.command.poses = [pose_msg]
            request.command.metric_names = [ig_mode]
            request.command.config.ray_resolution_x = 1
            request.command.config.ray_resolution_y = 1
            request.command.config.ray_window.min_x_perc = 0
            request.command.config.ray_window.min_y_perc = 0
            request.command.config.ray_window.max_x_perc = 1
            request.command.config.ray_window.max_y_perc = 1
            request.command.config.max_ray_depth = raycast_max_range
            with timer.measure("compute_ig_service"):
                response = compute_ig_service(request)
            assert response.expected_information[0].status == 0
            predicted_depth_image = None
            return response.expected_information[0].predicted_gain, predicted_depth_image

        def convert_pose_to_ros_transform_msg(pose):
            transform_msg = TransformMsg()
            ros_utils.point_numpy_to_ros(pose.location(), transform_msg.translation)
            orientation_quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
            ros_utils.quaternion_numpy_to_ros(orientation_quat, transform_msg.rotation)
            return transform_msg

        def insert_rpg_ig_depth_map(pose, depth_image, intrinsics):
            logger.info("Inserting depth map into RPG IG octomap")
            request = DepthMapInputRequest()
            transform_msg = convert_pose_to_ros_transform_msg(pose)
            request.sensor_to_world = transform_msg
            request.height = depth_image.shape[0]
            request.width = depth_image.shape[1]
            request.stride = request.width
            request.depths = np.asarray(depth_image.flatten(), dtype=np.float32)
            request.focal_length_x = intrinsics[0, 0]
            request.focal_length_y = intrinsics[1, 1]
            request.principal_point_x = intrinsics[0, 2]
            request.principal_point_y = intrinsics[1, 2]
            response = depth_map_input_service(request)
            assert response.success

        def reset_rpg_ig_octomap():
            logger.info("Resetting RPG IG octomap")
            request = ResetOctomapRequest()
            response = reset_octomap_service(request)
            assert response.success
            if args.limit_rpg_ig_to_score_bbox:
                score_bounding_box = environment.base.get_score_bounding_box()
                request2 = SetScoreBoundingBoxRequest()
                ros_utils.point_numpy_to_ros(score_bounding_box.minimum(), request2.min)
                ros_utils.point_numpy_to_ros(score_bounding_box.maximum(), request2.max)
                response2 = set_score_bounding_box_service(request2)
                assert response2.success

        return compute_rpg_ig_reward, insert_rpg_ig_depth_map, reset_rpg_ig_octomap

    # Check if RPG IG service offer IG mode
    def ensure_rpg_ig_mode_is_available(ig_mode):
        import rospy
        from ig_active_reconstruction_msgs.srv import StringList, StringListRequest
        ig_list_service = rospy.ServiceProxy("world/ig_list", StringList)
        request = StringListRequest()
        response = ig_list_service(request)
        if ig_mode not in response.names:
            logger.info("RPG IG mode {} is not available. Available modes: {}".format(ig_mode, response.names))
            raise RuntimeError("RPG IG mode is not available")

    if args.visualize:
        plot = DynamicPlot(rows=4)
        plot.get_axes(0).set_ylabel("Score")
        plot.get_axes(1).set_ylabel("Reward")
        plot.get_axes(2).set_ylabel("Error")
        plot.get_axes(3).set_ylabel("Relative error")
        plot.get_axes(3).set_xlabel("Step #")
        plot.show(block=False)
    else:
        plot = None

    measurement_hook = None
    reset_hook = None
    if args.policy_mode == "prediction":
        logger.info("Using prediction to compute reward")
        compute_reward_fn = compute_predicted_reward
        pass
    elif args.policy_mode == "action_prediction":
        logger.info("Using action prediction to compute reward")
        compute_reward_fn = compute_action_prediction_reward
        pass
    elif args.policy_mode == "heuristic":
        logger.info("WARNING: Using heuristic to compute reward")
        compute_reward_fn = compute_heuristic_reward
    elif args.policy_mode == "oracle":
        logger.info("WARNING: Using oracle to compute reward")
        compute_reward_fn = compute_oracle_reward
    elif args.policy_mode == "blind_oracle":
        logger.info("WARNING: Using blind oracle to compute reward")
        compute_reward_fn = compute_blind_oracle_reward
    elif args.policy_mode == "uniform":
        logger.info("WARNING: Using uniform (1) reward for random policy")
        compute_reward_fn = compute_uniform_reward
    elif args.policy_mode == "random":
        logger.info("WARNING: Using zero reward for random policy")
        compute_reward_fn = compute_zero_reward
    elif args.policy_mode.startswith("rpg_ig_"):
        logger.info("WARNING: Using RPG IG to compute ")
        ig_mode = args.policy_mode[len("rpg_ig_"):]
        ensure_rpg_ig_mode_is_available(ig_mode)
        rpg_ig_functions = rpg_ig_function_factory(ig_mode)
        compute_reward_fn = rpg_ig_functions[0]
        measurement_hook = rpg_ig_functions[1]
        reset_hook = rpg_ig_functions[2]
    else:
        logger.info("mode: {}".format(args.policy_mode))
        logger.info(args.policy_mode.startswith("rpg_ig_"))
        raise NotImplementedError("Unknown policy mode was specified: {}".format(args.policy_mode))

    file_num_arr = [0]

    def before_reset_hook(env):
        # return env.base.Pose([-12.0609442441, 14.1686405798,5.63435488626], [0.0, 0.0, 0.0809092196054])
        # env.base.set_prng(np.random.RandomState(42))
        rng = np.random.RandomState(prng_seed)
        new_seed = rng.randint(np.iinfo(np.int32).max)
        file_num_arr[0] = 0
        for i in range(file_num_arr[0]):
            new_seed = rng.randint(np.iinfo(np.int32).max)
        logger.info("Resetting environment for file_num {} with prng seed: {}".format(file_num_arr[0], new_seed))
        import time
        time.sleep(2)
        env.base.set_prng(np.random.RandomState(new_seed))
        # return environment.base.Pose([15.71507, -17.92461, 1.80241], [0.0, 0.0, 4.19036100111])
        return None

    def after_reset_hook(env):
        logger.info("Env reset in pose {}".format(env.base.get_pose()))
        import time
        time.sleep(5)

    environment.base.before_reset_hooks.register(before_reset_hook, environment)
    environment.base.after_reset_hooks.register(after_reset_hook, environment)

    episode = 0
    while args.num_episodes < 0 or episode < args.num_episodes:
        if args.output_filename_prefix:
            if args.plan_steps > 1:
                output_filename_template = "{:s}_{:s}_stereo_{:s}_{:.2f}_{:d}step_{{:d}}.hdf5".format(
                    args.output_filename_prefix,
                    args.policy_mode,
                    stereo_method,
                    stereo_baseline,
                    args.plan_steps)
            else:
                output_filename_template = "{:s}_{:s}_stereo_{:s}_{:.2f}_{{:d}}.hdf5".format(
                    args.output_filename_prefix,
                    args.policy_mode,
                    stereo_method,
                    stereo_baseline)
            hdf5_filename, file_num = file_utils.get_next_filename(output_filename_template)
            if not args.dry_run:
                file_num_arr[0] = file_num
            logger.info("File name for episode {}: {}".format(file_num, hdf5_filename))
            if args.max_file_num is not None and file_num + 1 > args.max_file_num:
                logger.info("Reached maximum file number. Stopping.")
                return
            logger.info("hdf5 filename: {}".format(hdf5_filename))
            if os.path.isfile(hdf5_filename):
                logger.info("File '{}' already exists. Skipping.".format(hdf5_filename))
                continue

        if plot is not None:
            plot.reset()

        logger.info("Running episode #{} out of {} with policy {}".format(episode, args.num_episodes, args.policy_mode))
        if reset_hook is not None:
            reset_hook()

        output = run_episode(args, episode, environment, compute_reward_fn, args.policy_mode,
                             args.reset_interval, args.reset_score_threshold,
                             downsample_to_grid=downsample_to_grid,
                             is_oracle=args.policy_mode == "oracle",
                             verbose=args.verbose, plot=plot,
                             visualize_images=args.visualize_images,
                             measurement_hook=measurement_hook,
                             measure_timing=args.measure_timing,
                             compute_predicted_reward=compute_predicted_reward)

        if plot is not None:
            import matplotlib.pyplot as plt
            plot_filename = "{:s}_{:s}_{:d}.png".format(
                args.output_filename_prefix,
                args.policy_mode,
                episode)
            if not args.dry_run:
                plt.savefig(plot_filename)
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
            if args.max_file_num is not None and file_num + 1 > args.max_file_num:
                logger.info("Reached maximum file number. Stopping.")
                return
        episode += 1

        if args.dry_run:
            file_num_arr[0] += 1


if __name__ == '__main__':
    np.set_printoptions(threshold=50)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbose', default=0, help='Set verbosity level.')
    parser.add_argument('--hdf5-data-stats-path', required=True, help='HDF5 data stats file.')
    parser.add_argument('--log-file', type=str)
    parser.add_argument('--dry-run', type=argparse_bool, default=False)
    parser.add_argument('--model-path', help='Model path.')
    parser.add_argument('--checkpoint', help='Checkpoint to restore.')
    parser.add_argument('--config', type=str, help='YAML configuration.')
    parser.add_argument('--model-config', type=str, help='YAML description of model.')
    parser.add_argument('--visualize', type=argparse_bool, default=False)
    parser.add_argument('--visualize-images', type=argparse_bool, default=False)
    parser.add_argument('--measure-timing', type=argparse_bool, default=False)
    parser.add_argument('--output-filename-prefix', type=str)
    parser.add_argument('--max-file-num', type=int,
                        help='Maximum id of output files after which to stop (also if already existing)')
    parser.add_argument('--interactive', type=argparse_bool, default=False)
    parser.add_argument('--ignore-collision', type=argparse_bool, default=False)
    parser.add_argument('--policy-mode', type=str, default="prediction",
                        help="Policy mode. One of 'prediction', 'heuristic', 'oracle' and 'random'")
    parser.add_argument('--reset-interval', default=100, type=int)
    parser.add_argument('--reset-score-threshold', type=float, default=0.3)
    parser.add_argument('--num-episodes', type=int, default=-1, help='Number of episodes. Default is infinite (-1).')
    parser.add_argument('--plan-steps', type=int, default=1, help='Number of steps to predict.')
    parser.add_argument('--limit-rpg-ig-to-score-bbox', type=argparse_bool, default=False)

    parser.add_argument('--obs-levels', default="0,1,2,3,4", type=str)
    parser.add_argument('--obs-sizes', default="16,16,16", type=str)
    parser.add_argument('--environment-config', type=str, required=True, help="Environment configuration file")
    parser.add_argument('--client-id', default=0, type=int)
    parser.add_argument('--prng-seed', type=int)

    # IO
    parser.add_argument('--io.timeout', type=int, default=10 * 60)

    # Resource allocation and Tensorflow configuration
    parser.add_argument('--tf.gpu_memory_fraction', type=float, default=0.75)
    parser.add_argument('--tf.intra_op_parallelism', type=int, default=1)
    parser.add_argument('--tf.inter_op_parallelism', type=int, default=4)
    parser.add_argument('--tf.log_device_placement', type=argparse_bool, default=False,
                        help="Report where operations are placed.")

    args = parser.parse_args()

    if args.plan_steps < 1:
        import sys
        sys.stderr.write("ERROR: Number of planning steps has to be greater than 0\n")
        parser.print_help()
        sys.exit(1)

    if args.policy_mode != "prediction" \
            and args.policy_mode != "action_prediction" \
            and args.policy_mode != "heuristic" \
            and args.policy_mode != "oracle" \
            and args.policy_mode != "blind_oracle" \
            and args.policy_mode != "uniform" \
            and args.policy_mode != "random" \
            and not args.policy_mode.startswith("rpg_ig_"):
        import sys
        sys.stderr.write("ERROR: Unknown policy mode was specified: {}\n".format(args.policy_mode))
        parser.print_help()
        sys.exit(1)

    if args.log_file is not None and len(args.log_file) > 0:
        import logging
        file_handler = logging.FileHandler(args.log_file)
        logger.addHandler(file_handler)

    run(args)
