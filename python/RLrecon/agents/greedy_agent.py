import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from agent import BaseAgent
from RLrecon import math_utils
from RLrecon import ros_utils
from RLrecon import utils


class GreedyAgent(BaseAgent):

    def __init__(self, environment, temperature=0.1,
                 ignore_unknown_voxels=False, use_surface_only=False):
        super(GreedyAgent, self).__init__(temperature)
        self._environment = environment
        self._ignore_unknown_voxels = ignore_unknown_voxels
        self._use_surface_only = use_surface_only
        self._uncertain_pc_pub = rospy.Publisher("uncertain_point_cloud", PointCloud2, queue_size=1)

    def _get_bbox_mask(self, point_cloud, bbox):
        within_bounding_box_mask = np.logical_and(
            np.logical_and(
                np.logical_and(
                    point_cloud['x'] >= bbox.minimum()[0],
                    point_cloud['x'] <= bbox.maximum()[0]
                ),
                np.logical_and(
                    point_cloud['y'] >= bbox.minimum()[1],
                    point_cloud['y'] <= bbox.maximum()[1]
                )
            ),
            np.logical_and(
                point_cloud['z'] >= bbox.minimum()[2],
                point_cloud['z'] <= bbox.maximum()[2]
            )
        )
        return within_bounding_box_mask

    def _get_surface_mask(self, point_cloud):
        surface_voxel_mask = point_cloud['is_surface']
        return surface_voxel_mask

    def _get_uncertain_mask(self, point_cloud):
        uncertain_voxel_mask = \
            np.logical_or(
                np.logical_not(point_cloud['is_known']),
                np.logical_and(
                    point_cloud['occupancy'] >= 0.35,
                    point_cloud['occupancy'] <= 0.75
                )
            )
        return uncertain_voxel_mask

    def _get_uncertain_surface_mask(self, point_cloud):
        voxel_mask = np.logical_or(
            self._get_uncertain_mask(point_cloud),
            self._get_surface_mask(point_cloud))
        return voxel_mask

    def _get_uncertain_surface_point_cloud(self, point_cloud_msg, bounding_box):
        pc = ros_utils.point_cloud2_ros_to_numpy(point_cloud_msg)
        pc = np.unique(pc)
        bbox_mask = self._get_bbox_mask(pc, bounding_box)
        pc = pc[bbox_mask]
        surface_mask = self._get_uncertain_surface_mask(pc)
        pc = pc[surface_mask]
        pc_xyz = ros_utils.structured_to_3d_array(pc)
        return pc_xyz

    def _get_uncertain_point_cloud(self, point_cloud_msg, bounding_box):
        pc = ros_utils.point_cloud2_ros_to_numpy(point_cloud_msg)
        pc = np.unique(pc)
        bbox_mask = self._get_bbox_mask(pc, bounding_box)
        pc = pc[bbox_mask]
        uncertain_mask = self._get_uncertain_mask(pc)
        pc = pc[uncertain_mask]
        pc_xyz = ros_utils.structured_to_3d_array(pc)
        return pc_xyz

    def _perform_raycast(self, state, max_range):
        width = self._environment._engine.get_width()
        height = self._environment._engine.get_height()
        focal_length = self._environment._engine.get_focal_length()
        ignore_unknown_voxels = self._ignore_unknown_voxels
        timer = utils.Timer()
        rr = self._environment._mapper.perform_raycast_camera_rpy(
            state.location(),
            state.orientation_rpy(),
            width, height, focal_length,
            ignore_unknown_voxels,
            max_range
        )
        print("Raycast took {} seconds".format(timer.elapsed_seconds()))
        return rr

    def _get_tentative_reward(self, state, action_index):
        if self._environment.is_action_allowed(state, action_index):
            tentative_state = self._environment.simulate_action(state, action_index)
            max_range = 14.0
            rr = self._perform_raycast(tentative_state, max_range)
            # Publish point cloud with uncertain voxels
            timer = utils.Timer()
            if self._use_surface_only:
                pc_xyz = self._get_uncertain_surface_point_cloud(rr.point_cloud, self._environment.get_bounding_box())
            else:
                pc_xyz = self._get_uncertain_point_cloud(rr.point_cloud, self._environment.get_bounding_box())
            print("Point cloud processing took {} seconds".format(timer.elapsed_seconds()))
            # # print("pc_xyz.shape: ", pc_xyz.shape)
            # uncertain_pc_msg = ros_utils.point_cloud2_numpy_to_ros(pc_xyz)
            # uncertain_pc_msg.header.frame_id = 'map'
            # self._uncertain_pc_pub.publish(uncertain_pc_msg)
            # Compute reward
            reward = pc_xyz.shape[0]
        else:
            reward = self._environment.get_action_not_allowed_reward()
        return reward

    def next_action(self, state):
        rewards = np.empty((self._environment.num_actions(),))
        for action_index in xrange(self._environment.num_actions()):
            print("Testing action {} [{}]".format(
                action_index,
                self._environment.get_action_name(action_index)))
            reward = self._get_tentative_reward(state, action_index)
            rewards[action_index] = reward
        # Choose a best action
        max_reward = np.max(rewards)
        if max_reward < 0:
            raise RuntimeError("Could not determine an allowed action")
        if max_reward <= 0:
            print("WARNING: Could not determine a useful action.")
        # Print reward for each action
        print("Rewards:")
        for action_index in xrange(self._environment.num_actions()):
            mark = "*" if rewards[action_index] == max_reward else ""
            print("  {} [{}] -> {} {}".format(
                action_index,
                self._environment.get_action_name(action_index),
                rewards[action_index],
                mark))
        # indices = np.arange(rewards.size)
        # best_indices = indices[rewards == best_reward]
        # rnd_index = np.random.randint(0, best_indices.size)
        # best_action_index = best_indices[rnd_index]
        # return best_action_index
            normalized_rewards = rewards / max_reward
        prob_dist = self._values_to_prob_distribution(normalized_rewards)
        print("Probabilities:")
        for action_index in xrange(self._environment.num_actions()):
            print("  {} [{}] -> {}".format(
                action_index,
                self._environment.get_action_name(action_index),
                prob_dist[action_index]))
        return prob_dist
