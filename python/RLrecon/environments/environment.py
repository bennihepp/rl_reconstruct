from __future__ import print_function

from mercurial.hg import update

import numpy as np
import math_utils
from utils import Timer


TERMINAL_SCORE_THRESHOLD = 0.75
DEFAULT_ACTION_PENALTY = -50.0
DEFAULT_ACTION_NOT_ALLOWED_REWARD = -10000.0
DEFAULT_ACTION_NOT_VALID_REWARD = -100.0


class BaseEnvironment(object):
    """BaseEnvironment represents the world in which an agent moves and the interactions.

    Here it combines a simulation engine (i.e. Unreal) with a mapper (i.e. OctomapExt server).
    The possible actions have to be given as an argument to the constructor.
    """

    class Pose(object):
        """Pose object represents the location and orientation of the agent"""
        def __init__(self, location, orientation_rpy):
            self._location = location
            self._orientation_rpy = orientation_rpy

        def location(self):
            return self._location

        def orientation_rpy(self):
            return self._orientation_rpy

        def __str__(self):
            return "xyz=({} {} {}), rpy=({} {} {})".format(
                self._location[0], self._location[1], self._location[2],
                self._orientation_rpy[0], self._orientation_rpy[1], self._orientation_rpy[2]
            )

    class Observation(object):
        """Observation represents the location and orientation of the agent and the local view of the occupancy map"""
        def __init__(self, location, orientation_rpy, occupancy_map):
            self._location = location
            self._orientation_rpy = orientation_rpy
            self._occupancy_map = occupancy_map

        def location(self):
            return self._location

        def orientation_rpy(self):
            return self._orientation_rpy

        def occupancy_map(self):
            return self._occupancy_map

        def __str__(self):
            return "Observation: xyz=({} {} {}), rpy=({} {} {}, map shape={}".format(
                self._location[0], self._location[1], self._location[2],
                self._orientation_rpy[0], self._orientation_rpy[1], self._orientation_rpy[2],
                self.occupancy_map.shape
            )

    def __init__(self,
                 world_bounding_box,
                 action_map,
                 update_map_flags=None,
                 action_rewards=None,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
        """Initialize base environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            action_map (list): Mapping from index to an action function.
            update_map_flags (list): Flags indicating whether the map should be updated on the corresponding action.
            action_rewards (list): Instantanious contributions to the reward for each action.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper (:obj:): Occupancy mapper (i.e. OctomapExt interface).
            clear_size (float): Size of bounding box to clear in the occupancy map on reset.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._minimum_pitch = 0.
        self._maximum_pitch = np.pi / 2.
        self._prev_score = 0.0
        self._world_bounding_box = world_bounding_box
        if engine is None:
            from engines.unreal_cv_wrapper import UnrealCVWrapper
            engine = UnrealCVWrapper()
        self._engine = engine
        if mapper is None:
            from mapping.octomap_ext_mapper import OctomapExtMapper
            mapper = OctomapExtMapper()
        self._mapper = mapper
        self._clear_size = clear_size
        self._action_map = action_map
        if update_map_flags is None:
            update_map_flags = [True] * len(action_map)
        self._update_map_flags = update_map_flags
        if action_rewards is None:
            action_rewards = [0.0] * len(action_map)
        self._action_rewards = action_rewards
        self._action_not_allowed_reward = action_not_allowed_reward
        self._action_not_valid_reward = DEFAULT_ACTION_NOT_VALID_REWARD
        self._filter_depth_map = filter_depth_map
        if use_ros:
            import rospy
            from geometry_msgs.msg import PoseStamped
            self._use_ros = True
            self._ros_world_frame = ros_world_frame
            self._pose_pub = rospy.Publisher(ros_pose_topic, PoseStamped, queue_size=10)
        else:
            self._use_ros = False
        self._score_bounding_box = score_bounding_box

    def _update_pose(self, new_pose):
        """Update pose and publish with ROS"""
        self._engine.set_location(new_pose._location)
        roll, pitch, yaw = new_pose._orientation_rpy
        self._engine.set_orientation_rpy(roll, pitch, yaw)
        self._publish_pose(new_pose)

    def _get_depth_point_cloud(self, pose):
        """Retrieve depth image point cloud in agent frame from simulation engine"""
        point_cloud = self._engine.get_depth_point_cloud_rpy(pose._location, pose._orientation_rpy,
                                                             filter=self._filter_depth_map)
        return point_cloud

    def _get_depth_point_cloud_world(self, pose):
        """Retrieve depth image point cloud in world frame from simulation engine"""
        point_cloud = self._engine.get_depth_point_cloud_world_rpy(pose._location, pose._orientation_rpy,
                                                                   filter=self._filter_depth_map)
        return point_cloud

    def _update_map(self, pose):
        """Update map by taking a depth image and integrating it into the occupancy map"""
        timer = Timer()
        point_cloud = self._get_depth_point_cloud(pose)
        t1 = timer.elapsed_seconds()
        result = self._mapper.update_map_rpy(pose._location, pose._orientation_rpy, point_cloud)
        t2 = timer.elapsed_seconds()
        print("Timing of _update_map():")
        print("  ", t1)
        print("  ", t2 - t1)
        print("Total: ", t2)
        return result.reward, result.normalized_score

    def _publish_pose(self, pose):
        """Publish pose if ROS is enabled"""
        if self._use_ros:
            import rospy
            from geometry_msgs.msg import PoseStamped
            import ros_utils
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self._ros_world_frame
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position = ros_utils.point_numpy_to_ros(pose.location())
            orientation_quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
            pose_msg.pose.orientation = ros_utils.quaternion_numpy_to_ros(orientation_quat)
            self._pose_pub.publish(pose_msg)

    def _get_observation(self, pose):
        """Return current observation of agent"""
        return pose

    def get_mapper(self):
        return self._mapper

    def get_world_bounding_box(self):
        """Get world bounding box"""
        return self._world_bounding_box

    def get_action_not_allowed_reward(self):
        """Return reward for invalid action"""
        return self._action_not_allowed_reward

    def get_pose(self):
        """Get current pose from simulation engine"""
        location = self._engine.get_location()
        orientation_rpy = self._engine.get_orientation_rpy()
        pose = self.Pose(location, orientation_rpy)
        self._publish_pose(pose)
        return pose

    def get_location(self):
        """Get current location from simulation engine"""
        location = self._engine.get_location()
        return location

    def get_orientation_rpy(self):
        """Get current rotation (roll, pitch, yaw) from simulation engine"""
        orientation_rpy = self._engine.get_orientation_rpy()
        return orientation_rpy

    def get_orientation_quat(self):
        """Get current rotation as a quaternion from simulation engine"""
        orientation_quat = self._engine.get_orientation_quat()
        return orientation_quat

    def set_pose(self, pose):
        """Set new pose"""
        self._update_pose(pose)

    def simulate_action_on_pose(self, pose, action_index):
        """Check if an action is allowed (i.e. no collision, inside bounding box)"""
        valid, new_pose = self._action_map[action_index](pose)
        return new_pose

    def is_action_allowed_on_pose(self, pose, action_index):
        location = pose.location()
        new_pose = self.simulate_action_on_pose(pose, action_index)
        new_location = new_pose.location()
        if np.any(new_location < self._world_bounding_box.minimum()):
            return False
        if np.any(new_location > self._world_bounding_box.maximum()):
            return False
        direction = new_location - location
        rays = [self._mapper.Ray(location, direction)]
        # TODO: Figure out what to do for collision safety. Here we just shoot a single ray.
        ignore_unknown_voxels = False
        rr = self._mapper.perform_raycast(rays, ignore_unknown_voxels)
        point_cloud = self._mapper.convert_raycast_point_cloud_from_msg(rr.point_cloud)
        hit_distance = np.linalg.norm(point_cloud[0].xyz - location)
        move_distance = np.linalg.norm(new_location - location)
        assert(rr.num_hits_occupied + rr.num_hits_unknown <= 1)
        if hit_distance <= move_distance:
            return False
        else:
            return True

    def reset(self, pose=None, reset_map=True):
        """Initialize environment (basically clears a bounding box in the occupancy map)"""
        self._prev_score = 0.0
        if pose is None:
            pose = self.get_pose()
        else:
            self._update_pose(pose)
        if reset_map:
            self._mapper.perform_reset()
        clear_bbox = math_utils.BoundingBox(
            pose.location() - self._clear_size / 2.0,
            pose.location() + self._clear_size / 2.0)
        bbox = math_utils.BoundingBox(
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([np.inf, np.inf, np.inf]))
        # bbox = math_utils.BoundingBox(
        #     np.array([-25, -25, -25]),
        #     np.array([+25, +25, +25]))
        if reset_map:
            # TODO: Currently clearing the whole map... Should only be free space
            self._mapper.perform_clear_bounding_box_voxels(bbox, densify=False)
            # self._mapper.perform_clear_bounding_box_voxels(bbox, densify=True)
            if self._score_bounding_box is not None:
                self._mapper.perform_override_bounding_box_voxels(self._score_bounding_box, 0.5)
            self._mapper.perform_clear_bounding_box_voxels(clear_bbox)
            # Only for debugging and visualization (RViz has problems showing free voxels)
            # self._mapper.perform_override_bounding_box_voxels(clear_bbox, 0.8)
        if self._score_bounding_box is not None:
            self._mapper.perform_set_score_bounding_box(self._score_bounding_box)

    def simulate_action_on_pose(self, pose, action_index):
        """Simulate the effect of an action on a pose"""
        new_pose = self._action_map[action_index](pose)
        return new_pose

    def num_actions(self):
        """Get total number of actions"""
        return len(self._action_map)

    def perform_action(self, action_index, pose=None):
        """Perform action by index"""
        # timer = Timer()
        if pose is None:
            pose = self.get_pose()
        else:
            self._update_pose(pose)
        # t1 = timer.elapsed_seconds()
        if not self.is_action_allowed_on_pose(pose, action_index):
            terminal = True
            info = {
                "score": self._prev_score,
            }
            return pose, self._action_not_allowed_reward, terminal, info
        # t2 = timer.elapsed_seconds()
        valid, new_pose = self._action_map[action_index](pose)
        if not valid:
            terminal = False
            info = {
                "score": self._prev_score,
            }
            return pose, self._action_not_valid_reward, terminal, info
        self._update_pose(new_pose)
        # t3 = timer.elapsed_seconds()
        if self._update_map_flags[action_index]:
            reward, score = self._update_map(new_pose)
        else:
            reward = 0
            score = self._prev_score
        reward += self._action_rewards[action_index]
        # t4 = timer.elapsed_seconds()
        observation = self._get_observation(new_pose)
        # t5 = timer.elapsed_seconds()
        # print("Timing of perform_action():")
        # print(t1)
        # print(t2 - t1)
        # print(t3 - t2)
        # print(t4 - t3)
        # print(t5 - t4)
        terminal = score >= TERMINAL_SCORE_THRESHOLD
        info = {
            "score": score,
        }
        self._prev_score = score
        return observation, reward, terminal, info

    def get_action_name(self, action_index):
        """Get method name by action index"""
        return self._action_map[action_index].__name__

    def _move(self, pose, offset):
        """Perform global motion of the agent"""
        new_location = pose.location() + offset
        valid = True
        return valid, self.Pose(new_location, pose.orientation_rpy())

    def _move_local(self, pose, local_offset):
        """Perform local motion in the agent frame"""
        quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
        world_offset = math_utils.rotate_vector_with_quaternion(quat, local_offset)
        valid = True
        return valid, self._move(pose, world_offset[:3])

    def _move_local_without_pr(self, pose, local_offset):
        """Perform local motion in the agent frame (overriding pitch and roll to 0)"""
        rpy_without_pitch_roll = np.array([0, 0, pose.orientation_rpy()[2]])
        quat_without_pitch_roll = math_utils.convert_rpy_to_quat(rpy_without_pitch_roll)
        world_offset = math_utils.rotate_vector_with_quaternion(quat_without_pitch_roll, local_offset)
        valid = True
        return valid, self._move(pose, world_offset[:3])

    def _rotate(self, pose, d_roll, d_pitch, d_yaw):
        """Perform rotation of the agent"""
        pitch = pose.orientation_rpy()[1]
        valid = True
        if d_pitch > 0:
            valid = pitch < self._maximum_pitch
        elif d_pitch < 0:
            valid = pitch > self._minimum_pitch
        new_rpy = pose.orientation_rpy() + np.array([d_roll, d_pitch, d_yaw])
        # print("New rpy: {}".format(new_rpy))
        if new_rpy[1] > self._maximum_pitch:
            new_rpy[1] = self._maximum_pitch
        elif new_rpy[1] < self._minimum_pitch:
            new_rpy[1] = self._minimum_pitch
        # print("Corrected new rpy: {}".format(new_rpy))
        return valid, self.Pose(pose.location(), new_rpy)

    # Only for debugging
    # def nop(self, pose):
    #     return self.Pose(pose.location(), pose.orientation_rpy())


class Environment(BaseEnvironment):
    """Environment adds local motions and rotations to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 move_distance=2.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 pitch_amount=math_utils.degrees_to_radians(180. / 5.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            move_distance (float): Scale of local motions.
            yaw_amount (float): Scale of yaw rotations.
            pitch_amount (float): Scale of pitch rotations.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._move_distance = move_distance
        self._yaw_amount = yaw_amount
        self._pitch_amount = pitch_amount
        action_map = [
            # self.nop,
            self.move_left,
            self.move_right,
            self.move_down,
            self.move_up,
            self.move_backward,
            self.move_forward,
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
            self.pitch_up,
            self.pitch_down,
        ]
        update_map_flags = [
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_map))
        super(Environment, self).__init__(
            world_bounding_box,
            action_map,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def move_left(self, pose):
        """Perform local move left"""
        return self._move_local_without_pr(pose, np.array([0, +self._move_distance, 0]))

    def move_right(self, pose):
        """Perform local move right"""
        return self._move_local_without_pr(pose, np.array([0, -self._move_distance, 0]))

    def move_down(self, pose):
        """Perform local move down"""
        return self._move_local_without_pr(pose, np.array([0, 0, -self._move_distance]))

    def move_up(self, pose):
        """Perform local move up"""
        return self._move_local_without_pr(pose, np.array([0, 0, +self._move_distance]))

    def move_backward(self, pose):
        """Perform local move backward"""
        return self._move_local_without_pr(pose, np.array([-self._move_distance, 0, 0]))

    def move_forward(self, pose):
        """Perform local move forward"""
        return self._move_local_without_pr(pose, np.array([+self._move_distance, 0, 0]))

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def pitch_up(self, pose):
        """Perform pitch rotation up"""
        return self._rotate(pose, 0, -self._pitch_amount, 0)

    def pitch_down(self, pose):
        """Perform pitch rotation down"""
        return self._rotate(pose, 0, +self._pitch_amount, 0)

    #TODO: Fix random placement
    def reset(self, **kwargs):
        """Resets the environment."""
        roll = 0
        pitch = 0
        if self._random_reset:
            yaw = 2 * np.pi * np.random.rand()
        else:
            yaw = 0
        valid_location = False
        center_location = 0.5 * (self.get_world_bounding_box().maximum() + self.get_world_bounding_box().minimum())
        location_range = 2 * (self.get_world_bounding_box().maximum() - self.get_world_bounding_box().minimum())
        min_location = center_location - 0.5 * location_range
        while not valid_location:
            location = min_location + np.random.rand(3) * location_range
            valid_location = np.linalg.norm(location) >= 8 and location[2] >= 4
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        super(Environment, self).reset(pose, **kwargs)

    # # TODO: Make proper collision detection
    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class SimpleV0Environment(BaseEnvironment):
    """SimpleV0Environment adds simple orbital motion actions to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
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
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        action_map = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_map))
        super(SimpleV0Environment, self).__init__(
            world_bounding_box,
            action_map,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_pose(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        roll = 0
        pitch = 0
        yaw = theta + np.pi
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        return pose

    def orbit_clockwise(self, pose):
        """Perform clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta - self._angle_amount
        new_pose = self._get_orbit_pose(new_theta)
        valid = True
        return valid, new_pose

    def orbit_counter_clockwise(self, pose):
        """Perform counter-clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta + self._angle_amount
        new_pose = self._get_orbit_pose(new_theta)
        valid = True
        return valid, new_pose

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
        else:
            theta = 0
        pose = self._get_orbit_pose(theta)
        super(SimpleV0Environment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class SimpleV1Environment(BaseEnvironment):
    """SimpleV1Environment adds simple orbital motion actions and yaw rotations to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
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
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        action_map = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise,
            self.orbit_clockwise_yaw_clockwise,
            self.orbit_clockwise_yaw_counter_clockwise,
            self.orbit_counter_clockwise_yaw_clockwise,
            self.orbit_counter_clockwise_yaw_counter_clockwise,
            # self.yaw_clockwise,
            # self.yaw_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True,
            True,
            True,
            True,
            True,
            # False,
            # False,
        ]
        action_rewards = np.array([
            -20.0,
            -20.0,
            -20.0,
            -20.0,
            -20.0,
            -20.0,
            # -1.0,
            # -1.0,
        ])
        super(SimpleV1Environment, self).__init__(
            world_bounding_box,
            action_map,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_location(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        return location

    def orbit_clockwise(self, pose):
        """Perform clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta - self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def orbit_counter_clockwise(self, pose):
        """Perform counter-clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta + self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def orbit_clockwise_yaw_clockwise(self, pose):
        valid1, pose1 = self.orbit_clockwise(pose)
        valid2, pose2 = self.yaw_clockwise(pose1)
        return valid1 and valid2, pose2

    def orbit_clockwise_yaw_counter_clockwise(self, pose):
        valid1, pose1 = self.orbit_clockwise(pose)
        valid2, pose2 = self.yaw_counter_clockwise(pose1)
        return valid1 and valid2, pose2

    def orbit_counter_clockwise_yaw_clockwise(self, pose):
        valid1, pose1 = self.orbit_counter_clockwise(pose)
        valid2, pose2 = self.yaw_clockwise(pose1)
        return valid1 and valid2, pose2

    def orbit_counter_clockwise_yaw_counter_clockwise(self, pose):
        valid1, pose1 = self.orbit_counter_clockwise(pose)
        valid2, pose2 = self.yaw_counter_clockwise(pose1)
        return valid1 and valid2, pose2

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
            if np.random.rand() < 0.25:
                yaw = 2 * np.pi * np.random.rand()
            else:
                d_yaw = np.pi / 4 * (np.random.rand() - 0.5)
                yaw = theta + np.pi + d_yaw
        else:
            theta = 0
            yaw = theta + np.pi
        location = self._get_orbit_location(theta)
        roll = 0
        pitch = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        super(SimpleV1Environment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class SimpleV2Environment(BaseEnvironment):
    """SimpleV2Environment adds simple orbital motion actions and yaw rotations to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 8.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 8.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
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
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        action_map = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise,
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True,
            False,
            False,
        ]
        action_rewards = np.array([
            -20.0,
            -20.0,
            -1.0,
            -1.0,
        ])
        super(SimpleV2Environment, self).__init__(
            world_bounding_box,
            action_map,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_location(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        return location

    def orbit_clockwise(self, pose):
        """Perform clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta - self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def orbit_counter_clockwise(self, pose):
        """Perform counter-clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta + self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
            if np.random.rand() < 0.25:
                yaw = 2 * np.pi * np.random.rand()
            else:
                d_yaw = np.pi / 4 * (np.random.rand() - 0.5)
                yaw = theta + np.pi + d_yaw
        else:
            theta = 0
            yaw = theta + np.pi
        location = self._get_orbit_location(theta)
        roll = 0
        pitch = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        super(SimpleV2Environment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class SimpleV3Environment(BaseEnvironment):
    """SimpleV3Environment adds simple orbital motion actions and yaw/pitch rotations to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 radius=15.0,
                 height=5.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 pitch_amount=math_utils.degrees_to_radians(180. / 5.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
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
            pitch_amount (float): Scale of pitch rotations.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        self._pitch_amount = pitch_amount
        action_map = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise,
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
            self.pitch_down,
            self.pitch_up,
        ]
        update_map_flags = [
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_map))
        super(SimpleV3Environment, self).__init__(
            world_bounding_box,
            action_map,
            update_map_flags=update_map_flags,
            action_rewards = action_rewards,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_location(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        return location

    def orbit_clockwise(self, pose):
        """Perform clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta - self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def orbit_counter_clockwise(self, pose):
        """Perform counter-clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta + self._angle_amount
        new_location = self._get_orbit_location(new_theta)
        new_pose = self.Pose(new_location, pose.orientation_rpy())
        valid = True
        return valid, new_pose

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def pitch_up(self, pose):
        """Perform pitch rotation up"""
        return self._rotate(pose, 0, -self._pitch_amount, 0)

    def pitch_down(self, pose):
        """Perform pitch rotation down"""
        return self._rotate(pose, 0, +self._pitch_amount, 0)

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
        location = self._get_orbit_location(theta)
        roll = 0
        pitch = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        super(SimpleV3Environment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class VerySimpleEnvironment(BaseEnvironment):
    """VerySimpleEnvironment adds simple orbital motion actions to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 engine=None,
                 mapper=None,
                 clear_size=6.0,
                 random_reset=True,
                 radius=15.0,
                 height=5.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 action_not_allowed_reward=-100.,
                 filter_depth_map=False,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map',
                 score_bounding_box=None):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_size (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
            use_ros (bool): Whether to use ROS and publish on some topics.
            ros_pose_topic (str): If ROS is used publish agent poses on this topic.
            ros_world_frame (str): If ROS is used this is the id of the world frame.
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        action_map = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise
        ]
        super(VerySimpleEnvironment, self).__init__(
            world_bounding_box,
            action_map,
            engine=engine,
            mapper=mapper,
            clear_size=clear_size,
            action_not_allowed_reward=action_not_allowed_reward,
            filter_depth_map=filter_depth_map,
            use_ros=use_ros,
            ros_pose_topic=ros_pose_topic,
            ros_world_frame=ros_world_frame,
            score_bounding_box=score_bounding_box)

    def _get_orbit_angle(self, pose):
        theta = np.arctan2(pose.location()[1], pose.location()[0])
        return theta

    def _get_orbit_pose(self, theta):
        x = self._radius * np.cos(theta)
        y = self._radius * np.sin(theta)
        z = self._height
        location = np.array([x, y, z])
        roll = 0
        pitch = 0
        yaw = theta + np.pi
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        return pose

    def orbit_clockwise(self, pose):
        """Perform clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta - self._angle_amount
        new_pose = self._get_orbit_pose(new_theta)
        valid = True
        return valid, new_pose

    def orbit_counter_clockwise(self, pose):
        """Perform counter-clockwise orbit move"""
        current_theta = self._get_orbit_angle(pose)
        new_theta = current_theta + self._angle_amount
        new_pose = self._get_orbit_pose(new_theta)
        valid = True
        return valid, new_pose

    def reset(self, **kwargs):
        """Resets the environment. Orbit angle is set to zero or randomly initialized."""
        # if pose is None:
        #     pose = self.get_pose()
        # theta = self._get_orbit_angle(pose)
        # pose = self._get_orbit_pose(theta)
        if self._random_reset:
            theta = 2 * np.pi * np.random.rand()
        else:
            theta = 0
        pose = self._get_orbit_pose(theta)
        super(VerySimpleEnvironment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True
