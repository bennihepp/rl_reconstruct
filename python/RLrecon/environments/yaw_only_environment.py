import numpy as np
from environment import BaseEnvironment
from RLrecon import math_utils


class YawOnlyEnvironmentV1(BaseEnvironment):

    def __init__(self,
                 world_bounding_box,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 6.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_size (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._yaw_amount = yaw_amount
        action_list = [
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True
        ]
        action_rewards = [
            -50.,
            -50.,
        ]
        self._obs_level = 3
        self._obs_size_x = 8
        self._obs_size_y = self._obs_size_x
        self._obs_size_z = self._obs_size_x
        super(YawOnlyEnvironmentV1, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            terminal_score_threshold=0.6,
            **kwargs)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_location(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        return location

    def get_observation_shapes(self):
        return [
            (self._obs_size_x, self._obs_size_y, self._obs_size_z, 2)
        ]

    def _get_observation(self, pose):
        level = self._obs_level
        size_x = self._obs_size_x
        size_y = self._obs_size_y
        size_z = self._obs_size_z
        center = self.get_location()
        orientation_rpy = self.get_orientation_rpy()
        # center = np.array([0, 0, 0])
        # orientation_rpy = np.array([0, 0, 0])
        # We query a subvolume of the occupancy map so that z-axis is aligned with gravity (roll = pitch = 0)
        # query_orientation_rpy = np.array([0, 0, orientation_rpy[2]])
        query_orientation_rpy = np.array([0, orientation_rpy[1], orientation_rpy[2]])
        # TODO: Should be exposed in environment
        res = self._mapper.perform_query_subvolume_rpy(
            center, query_orientation_rpy, level, size_x, size_y, size_z)
        occupancies = np.asarray(res.occupancies, dtype=np.float32)
        occupancies_3d = np.reshape(occupancies, (size_x, size_y, size_z))
        observation_certainties = np.asarray(res.observation_certainties, dtype=np.float32)
        observation_certainties_3d = np.reshape(observation_certainties, (size_x, size_y, size_z))
        grid_3d = np.stack([occupancies_3d, observation_certainties_3d], axis=-1)
        return [grid_3d]
        # location = self.get_location()
        # # orientation_rpy = self.get_orientation_rpy()
        # orientation_quat = self.get_orientation_quat()
        # # return [location, orientation_quat, occupancies_3d]
        # # return [location, orientation_quat, grid_3d]
        # previous_state_orientation_quat = math_utils.convert_rpy_to_quat(self._previous_state.orientation_rpy())
        # orientation_quat *= np.sign(orientation_quat[3])
        # previous_state_orientation_quat *= np.sign(previous_state_orientation_quat[3])
        # return [location, orientation_quat, self._previous_state.location(), previous_state_orientation_quat]

    def perform_action(self, action_index, pose=None):
        observation, reward, terminal, info = super(YawOnlyEnvironmentV1, self).perform_action(action_index, pose)
        if reward > 0:
            terminal = True
            reward = 500.
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
            yaw = 2 * np.pi * np.random.rand()
        else:
            theta = 0
            yaw = theta + np.pi
        yaw = theta + np.pi
        if np.random.rand() > 0.5:
            yaw += 2 * self._yaw_amount
        else:
            yaw -= 2 * self._yaw_amount
        location = self._get_orbit_location(theta)
        roll = 0
        pitch = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        return super(YawOnlyEnvironmentV1, self).reset(pose, **kwargs)

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class YawOnlyEnvironmentV2(BaseEnvironment):

    def __init__(self,
                 world_bounding_box,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 6.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_size (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._yaw_amount = yaw_amount
        action_list = [
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True
        ]
        action_rewards = [
            -200.,
            -200.,
        ]
        self._obs_level = 3
        self._obs_size_x = 8
        self._obs_size_y = self._obs_size_x
        self._obs_size_z = self._obs_size_x
        super(YawOnlyEnvironmentV2, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            terminal_score_threshold=0.6,
            **kwargs)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_location(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        return location

    def get_observation_shapes(self):
        return [
            (self._obs_size_x, self._obs_size_y, self._obs_size_z, 2)
        ]

    def _get_observation(self, pose):
        level = self._obs_level
        size_x = self._obs_size_x
        size_y = self._obs_size_y
        size_z = self._obs_size_z
        center = self.get_location()
        orientation_rpy = self.get_orientation_rpy()
        # center = np.array([0, 0, 0])
        # orientation_rpy = np.array([0, 0, 0])
        # We query a subvolume of the occupancy map so that z-axis is aligned with gravity (roll = pitch = 0)
        # query_orientation_rpy = np.array([0, 0, orientation_rpy[2]])
        query_orientation_rpy = np.array([0, orientation_rpy[1], orientation_rpy[2]])
        # TODO: Should be exposed in environment
        res = self._mapper.perform_query_subvolume_rpy(
            center, query_orientation_rpy, level, size_x, size_y, size_z)
        occupancies = np.asarray(res.occupancies, dtype=np.float32)
        occupancies_3d = np.reshape(occupancies, (size_x, size_y, size_z))
        observation_certainties = np.asarray(res.observation_certainties, dtype=np.float32)
        observation_certainties_3d = np.reshape(observation_certainties, (size_x, size_y, size_z))
        grid_3d = np.stack([occupancies_3d, observation_certainties_3d], axis=-1)
        return [grid_3d]
        # location = self.get_location()
        # # orientation_rpy = self.get_orientation_rpy()
        # orientation_quat = self.get_orientation_quat()
        # # return [location, orientation_quat, occupancies_3d]
        # # return [location, orientation_quat, grid_3d]
        # previous_state_orientation_quat = math_utils.convert_rpy_to_quat(self._previous_state.orientation_rpy())
        # orientation_quat *= np.sign(orientation_quat[3])
        # previous_state_orientation_quat *= np.sign(previous_state_orientation_quat[3])
        # return [location, orientation_quat, self._previous_state.location(), previous_state_orientation_quat]

    def perform_action(self, action_index, pose=None):
        observation, reward, terminal, info = super(YawOnlyEnvironmentV2, self).perform_action(action_index, pose)
        if reward > self._action_rewards[action_index]:
            terminal = True
            reward = 100.
        return observation, reward, terminal, info

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
            yaw = 2 * np.pi * np.random.rand()
        else:
            theta = 0
            yaw = theta + np.pi
        yaw = theta + np.pi
        if np.random.rand() > 0.5:
            yaw += 5 * self._yaw_amount
        else:
            yaw -= 5 * self._yaw_amount
        location = self._get_orbit_location(theta)
        roll = 0
        pitch = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        return super(YawOnlyEnvironmentV2, self).reset(pose, **kwargs)

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True
