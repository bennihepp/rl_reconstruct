from __future__ import print_function
import numpy as np
from pybh import math_utils, utils


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
                 action_list,
                 update_map_flags=None,
                 action_rewards=None,
                 engine=None,
                 mapper=None,
                 clear_extent=6.0,
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 action_not_valid_reward=DEFAULT_ACTION_NOT_VALID_REWARD,
                 terminal_score_threshold=TERMINAL_SCORE_THRESHOLD,
                 filter_depth_map=False,
                 start_bounding_box=None,
                 score_bounding_box=None,
                 collision_obs_level=1,
                 collision_obs_sizes=None,
                 collision_bbox=None,
                 collision_bbox_obs_level=0,
                 collision_occupancy_threshold=0.3,
                 collision_observation_certainty_threshold=0.1,
                 collision_check_above_camera=False,
                 collision_check_above_camera_distance=10,
                 prng_or_seed=None):
        """Initialize base environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            action_list (list): Mapping from index to an action function.
            update_map_flags (list): Flags indicating whether the map should be updated on the corresponding action.
            action_rewards (list): Instantanious contributions to the reward for each action.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper (:obj:): Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """
        if isinstance(prng_or_seed, np.random.RandomState):
            self._prng = prng_or_seed
        else:
            self._prng = np.random.RandomState(prng_or_seed)

        self._collision_occupancy_threshold = collision_occupancy_threshold
        self._collision_observation_certainty_threshold = collision_observation_certainty_threshold
        # TODO: Kind of hacky. Prevents reset position inside of buildings or structures.
        self._collision_check_above_camera = collision_check_above_camera
        self._collision_check_above_camera_distance = collision_check_above_camera_distance

        self._before_reset_hooks = utils.Callbacks(return_first_value=True)
        self._after_reset_hooks = utils.Callbacks()
        self._update_pose_hooks = utils.Callbacks()

        self._observation_count_saturation = 3.0
        self._minimum_pitch = 0.
        self._maximum_pitch = np.pi / 2.
        self._prev_score = 0.0
        self._world_bounding_box = world_bounding_box
        if engine is None:
            from pybh.unreal.unreal_cv_wrapper import UnrealCVWrapper
            engine = UnrealCVWrapper()
        self._engine = engine
        if mapper is None:
            from RLrecon.mapping.octomap_ext_mapper import OctomapExtMapper
            mapper = OctomapExtMapper()
        mapper.perform_reset(reset_stack=True, reset_storage=True)
        self._mapper = mapper
        clear_extent = np.array(clear_extent)
        if len(clear_extent) == 1:
            clear_extent = np.array([clear_extent, clear_extent, clear_extent])
        self._clear_extent = clear_extent
        self._action_list = action_list
        if update_map_flags is None:
            update_map_flags = [True] * len(action_list)
        self._update_map_flags = update_map_flags
        if action_rewards is None:
            action_rewards = [0.0] * len(action_list)
        self._action_rewards = action_rewards
        self._action_not_allowed_reward = action_not_allowed_reward
        self._action_not_valid_reward = action_not_valid_reward
        self._terminal_score_threshold = terminal_score_threshold
        self._filter_depth_map = filter_depth_map
        self._score_bounding_box = score_bounding_box
        if self._score_bounding_box is None:
            self._score_bounding_box = self._world_bounding_box
        self._start_bounding_box = start_bounding_box
        if self._start_bounding_box is None:
            self._start_bounding_box = self._world_bounding_box
        assert self._world_bounding_box.contains(self._start_bounding_box), \
            "Start bounding box must be fully inside of world bounding box"
        self._collision_obs_level = collision_obs_level
        if collision_obs_sizes is None:
            collision_obs_sizes = [3, 3, 3]
        self._collision_obs_sizes = collision_obs_sizes
        if collision_bbox is None:
            map_resolution = self.get_mapper().perform_info().map_resolution
            collision_bbox_extent = np.array([3 * map_resolution] * 3)
            collision_bbox = math_utils.BoundingBox(-0.5 * collision_bbox_extent, +0.5 * collision_bbox_extent)
        self._collision_bbox = collision_bbox
        self._collision_bbox_obs_level = collision_bbox_obs_level

    def set_prng(self, prng):
        """Overwrite pseudo-random number generator. This is useful for evaulation purposes."""
        self._prng = prng

    @property
    def before_reset_hooks(self):
        return self._before_reset_hooks

    @property
    def after_reset_hooks(self):
        return self._after_reset_hooks

    @property
    def update_pose_hooks(self):
        return self._update_pose_hooks

    def _update_pose(self, new_pose, wait_until_set=False, broadcast=True):
        """Update pose and publish with ROS"""
        # self._engine.set_location(new_pose.location())
        # roll, pitch, yaw = new_pose.orientation_rpy()
        # self._engine.set_orientation_rpy(roll, pitch, yaw)
        self._engine.set_pose_rpy((new_pose.location(), new_pose.orientation_rpy()), wait_until_set)
        if broadcast:
            self._update_pose_hooks(new_pose)

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
        # # timer = Timer()
        # point_cloud = self._get_depth_point_cloud(pose)
        # # t1 = timer.elapsed_seconds()
        # result = self._mapper.perform_insert_point_cloud_rpy(pose.location(), pose.orientation_rpy(), point_cloud)
        # # t2 = timer.elapsed_seconds()

        self._engine.set_pose_rpy(pose, wait_until_set=True)
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.get_focal_length()
        intrinsics[1, 1] = intrinsics[0, 0]
        intrinsics[0, 2] = float(self._engine.get_width()) / 2
        intrinsics[1, 2] = float(self._engine.get_height()) / 2
        depth_image = self._engine.get_depth_image()
        downsample_to_grid = True
        simulate = False
        result = self._mapper.perform_insert_depth_map_rpy(pose.location(), pose.orientation_rpy(),
                                                           depth_image, intrinsics, downsample_to_grid, simulate)

        # print("Timing of _update_map():")
        # print("  ", t1)
        # print("  ", t2 - t1)
        # print("Total: ", t2)
        return result.reward, result.normalized_score

    def _get_observation(self, pose):
        """Return current observation of agent"""
        return pose

    def get_engine(self):
        return self._engine

    def get_mapper(self):
        return self._mapper

    def get_world_bounding_box(self):
        """Get world bounding box"""
        return self._world_bounding_box

    def get_score_bounding_box(self):
        """Get score bounding box"""
        return self._score_bounding_box

    def get_start_bounding_box(self):
        """Get start bounding box"""
        return self._start_bounding_box

    def get_collision_bounding_box(self):
        """Get collision bounding box"""
        return self._collision_bbox

    def get_action_not_allowed_reward(self):
        """Return reward for invalid action"""
        return self._action_not_allowed_reward

    def get_pose(self):
        """Get current pose from simulation engine"""
        location, orientation_rpy = self._engine.get_pose_rpy()
        pose = self.Pose(location, orientation_rpy)
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

    def set_pose(self, pose, wait_until_set=False, broadcast=True):
        """Set new pose"""
        self._update_pose(pose, wait_until_set, broadcast)

    def simulate_action_on_pose(self, pose, action_index):
        """Check if an action is allowed (i.e. no collision, inside bounding box)"""
        if action_index == -1:
            # NOP action
            return pose
        valid, new_pose = self._action_list[action_index](pose)
        return new_pose

    def is_pose_colliding(self, pose, verbose=False):
        location = pose.location()
        if np.any(location < self._world_bounding_box.minimum()):
            if verbose:
                print("World bbox collision")
            return True
        if np.any(location > self._world_bounding_box.maximum()):
            if verbose:
                print("World bbox collision")
            return True

        # TODO: Decide which variant to use and remove the other one
        # if np.all(location >= self._score_bounding_box.minimum()) \
        #         and np.all(location <= self._score_bounding_box.maximum()):
        if True:
            bbox = self._collision_bbox.move(location)
            res = self.get_mapper().perform_query_bbox(bbox, self._collision_bbox_obs_level, dense=True)
            # res = self.get_mapper().perform_query_subvolume_rpy(
            #     location, pose.orientation_rpy(),
            #     self._collision_obs_level,s
            #     self._collision_obs_sizes[0], self._collision_obs_sizes[1], self._collision_obs_sizes[2],
            #     axis_mode=0)
            if len(res.occupancies) > 0:
                occupancies = np.asarray(res.occupancies)
                observation_certainties = np.asarray(res.observation_certainties)
                if np.any(occupancies > self._collision_occupancy_threshold):
                    if verbose:
                        print("Collision because of occupancies")
                        print(occupancies)
                    return True
                if np.any(observation_certainties < self._collision_observation_certainty_threshold):
                    if verbose:
                        print("Collision because of observation certainties")
                    return True

        # No collision
        return False

    def is_ray_colliding(self, ray_origin, ray_target=None, ignore_unknown_voxels=False, verbose=False):
        if ray_target is None:
            assert isinstance(ray_origin, self._mapper.Ray)
            ray = ray_origin
        else:
            ray = self._mapper.Ray(ray_origin, ray_target - ray_origin)
        rays = [ray]
        max_range = -1
        rr = self._mapper.perform_raycast(rays, ignore_unknown_voxels=ignore_unknown_voxels,
                                          max_range=max_range, only_in_score_bounding_box=False)
        point_cloud = self._mapper.convert_raycast_point_cloud_from_msg(rr.point_cloud)
        hit_distance = np.linalg.norm(point_cloud[0].xyz - ray_origin)
        target_distance = np.linalg.norm(ray_target - ray_origin)
        assert rr.num_hits_occupied + rr.num_hits_unknown <= 1
        if hit_distance <= target_distance:
            if verbose:
                print("Collision because of raycast")
                print("hit_distance:", hit_distance)
                print("target_distance:", target_distance)
                print("ray_origin:", ray_origin)
                print("xyz:", point_cloud[0].xyz)
                print("occupancy:", point_cloud[0].occupancy)
                print("observation_certainty:", point_cloud[0].observation_certainty)
                print("is_surface:", point_cloud[0].is_surface)
                print("is_known:", point_cloud[0].is_known)
            return True
        else:
            return False

    def is_action_colliding(self, pose, action_index, ignore_unknown_voxels=False, verbose=False):
        location = pose.location()
        new_pose = self.simulate_action_on_pose(pose, action_index)
        new_location = new_pose.location()

        if np.allclose(location, new_location):
            # Pure rotations are always allowed.
            # If you start from a colliding position this will still pass.
            # if verbose:
            #     print("Pure rotation -> no collision")
            return False

        if self.is_pose_colliding(new_pose, verbose=verbose):
            return True

        # TODO: Figure out what to do for collision safety. Here we just shoot a single ray.
        if self.is_ray_colliding(location, new_location, ignore_unknown_voxels=ignore_unknown_voxels, verbose=verbose):
            return True
        else:
            return False

    def is_action_allowed_on_pose(self, pose, action_index):
        return self.is_action_colliding(pose, action_index)

    def _reset(self, pose=None, reset_map=True, keep_pose=False):
        self._prev_score = 0.0
        if keep_pose or pose is None:
            pose = self.get_pose()
        else:
            self._update_pose(pose)
        if reset_map:
            self._mapper.perform_reset()
            if np.any(self._clear_extent > 0):
                clear_bbox = math_utils.BoundingBox(
                    - self._clear_extent / 2.0,
                    + self._clear_extent / 2.0).move(pose.location())
                clear_occupancy = 0.0
                clear_observation_count = self._observation_count_saturation
                self._mapper.perform_override_bounding_box_voxels(
                    clear_bbox, clear_occupancy, clear_observation_count, densify=False)
            # Only for debugging and visualization (RViz has problems showing free voxels)
            # self._mapper.perform_override_bounding_box_voxels(clear_bbox, 0.8, self._observation_count_saturation)
        if self._score_bounding_box is not None:
            self._mapper.perform_set_score_bounding_box(self._score_bounding_box)
        observation = self._get_observation(pose)
        return observation

    def reset(self, pose=None, reset_map=True, keep_pose=False):
        """Initialize environment (basically clears a bounding box in the occupancy map)"""
        self._before_reset_hooks()
        observation = self._reset(pose, reset_map, keep_pose)
        self._after_reset_hooks()
        return observation

    def get_num_of_actions(self):
        """Get total number of actions"""
        return len(self._action_list)

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
        valid, new_pose = self._action_list[action_index](pose)
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
        terminal = score >= self._terminal_score_threshold
        info = {
            "score": score,
        }
        self._prev_score = score
        return observation, reward, terminal, info

    def get_action_name(self, action_index):
        """Get method name by action index"""
        return self._action_list[action_index].__name__

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
        return self._move(pose, world_offset[:3])

    def _move_local_without_pr(self, pose, local_offset):
        """Perform local motion in the agent frame (overriding pitch and roll to 0)"""
        rpy_without_pitch_roll = np.array([0, 0, pose.orientation_rpy()[2]])
        quat_without_pitch_roll = math_utils.convert_rpy_to_quat(rpy_without_pitch_roll)
        world_offset = math_utils.rotate_vector_with_quaternion(quat_without_pitch_roll, local_offset)
        return self._move(pose, world_offset[:3])

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
                 random_reset=True,
                 move_distance=2.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 pitch_amount=math_utils.degrees_to_radians(180. / 5.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            move_distance (float): Scale of local motions.
            yaw_amount (float): Scale of yaw rotations.
            pitch_amount (float): Scale of pitch rotations.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._move_distance = move_distance
        self._yaw_amount = yaw_amount
        self._pitch_amount = pitch_amount
        action_list = [
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
        action_rewards = np.array([action_penalty] * len(action_list))
        super(Environment, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            **kwargs)

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
            yaw = 2 * np.pi * self._prng.rand()
        else:
            yaw = 0
        valid_location = False
        center_location = 0.5 * (self.get_start_bounding_box().maximum() + self.get_start_bounding_box().minimum())
        location_range = 2 * (self.get_start_bounding_box().maximum() - self.get_start_bounding_box().minimum())
        min_location = center_location - 0.5 * location_range
        while not valid_location:
            location = min_location + self._prng.rand(3) * location_range
            valid_location = np.linalg.norm(location) >= 8 and location[2] >= 4
        orientation_rpy = np.array([roll, pitch, yaw])
        pose = self.Pose(location, orientation_rpy)
        super(Environment, self).reset(pose, **kwargs)

    # # TODO: Make proper collision detection
    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class HorizontalEnvironment(Environment):

    def __init__(self,
                 world_bounding_box,
                 random_reset=True,
                 move_distance=2.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount_degrees=None,
                 # pitch_amount=math_utils.degrees_to_radians(180. / 5.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            move_distance (float or np.ndarray((3,)): Scale of local motions (along each axis).
            y_move_distance (float): Scale of local motions in y axis.
            z_move_distance (float): Scale of local motions in z axis.
            yaw_amount (float): Scale of yaw rotations.
            yaw_amount_degrees (float: Scale of yaw rotation in degrees. Overwrites yaw_amount if not `None`.
            pitch_amount (float): Scale of pitch rotations.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        move_distance = np.array(move_distance)
        if move_distance.ndim == 0:
            move_distance = np.array([move_distance, move_distance, move_distance])
        assert(len(move_distance) == 3)
        self._move_distance = move_distance
        self._yaw_amount = yaw_amount
        if yaw_amount_degrees is not None:
            self._yaw_amount = math_utils.degrees_to_radians(yaw_amount_degrees)
        # self._pitch_amount = pitch_amount
        action_list = [
            # self.nop,
            self.move_left,
            self.move_right,
            # self.move_down,
            # self.move_up,
            self.move_backward,
            self.move_forward,
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
            self.yaw_turn_around,
            # self.pitch_up,
            # self.pitch_down,
        ]
        update_map_flags = [
            True,
            True,
            # True,
            # True,
            False,
            False,
            False,
            False,
            False,
            # False,
            # False,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_list))
        super(Environment, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            **kwargs)

    def _get_origin_angle(self, location):
        theta = np.arctan2(location[1], location[0])
        return theta

    def move_left(self, pose):
        """Perform local move left"""
        return self._move_local_without_pr(pose, np.array([0, +self._move_distance[1], 0]))

    def move_right(self, pose):
        """Perform local move right"""
        return self._move_local_without_pr(pose, np.array([0, -self._move_distance[1], 0]))

    def move_down(self, pose):
        """Perform local move down"""
        return self._move_local_without_pr(pose, np.array([0, 0, -self._move_distance[2]]))

    def move_up(self, pose):
        """Perform local move up"""
        return self._move_local_without_pr(pose, np.array([0, 0, +self._move_distance[2]]))

    def move_backward(self, pose):
        """Perform local move backward"""
        return self._move_local_without_pr(pose, np.array([-self._move_distance[0], 0, 0]))

    def move_forward(self, pose):
        """Perform local move forward"""
        return self._move_local_without_pr(pose, np.array([+self._move_distance[0], 0, 0]))

    def yaw_clockwise(self, pose):
        """Perform yaw rotation clockwise"""
        return self._rotate(pose, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, pose):
        """Perform yaw rotation counter-clockwise"""
        return self._rotate(pose, 0, 0, +self._yaw_amount)

    def yaw_turn_around(self, pose):
        """Perform yaw rotation around 180 degrees"""
        return self._rotate(pose, 0, 0, np.pi)

    #TODO: Fix random placement
    def reset(self, ignore_collision=False, pose=None, **kwargs):
        """Resets the environment."""
        override_pose = self._before_reset_hooks()
        if override_pose is not None:
            pose = override_pose
        if pose is not None:
            reset_result = super(Environment, self)._reset(pose, **kwargs)
            # Check if all actions are still possible after reset
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    print("WARNING: Sampled location collides after map reset. Adjust the clearing bbox")
                for action_index in range(self.get_num_of_actions()):
                    if self.is_action_colliding(pose, action_index, verbose=True):
                        print("WARNING: Sampled location collides for action {} after map reset. "
                              "Adjust the clearing bbox".format(action_index))
                        break
            self._after_reset_hooks()
            return reset_result
        roll = 0
        pitch = 0
        if self._random_reset:
            # yaw = 2 * np.pi * self._prng.rand()
            u = self._prng.rand()
            yaw = 2 * np.pi * u
        else:
            yaw = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        center_location = 0.5 * (self.get_start_bounding_box().maximum() + self.get_start_bounding_box().minimum())
        location_range = self.get_start_bounding_box().maximum() - self.get_start_bounding_box().minimum()
        min_location = center_location - 0.5 * location_range
        valid_location = False
        reset_necessary = True
        while not valid_location:
            if reset_necessary:
                cleared_world_bbox_with_surface_voxels_key = "cleared_world_bbox_with_surface_voxels"
                if self.get_mapper().does_octomap_exist(cleared_world_bbox_with_surface_voxels_key):
                    print("Loading cleared octomap with surface voxels")
                    self.get_mapper().restore_octomap(cleared_world_bbox_with_surface_voxels_key)
                    reset_necessary = False
                else:
                    print("Map reset")
                    self.get_mapper().perform_reset()
                    clear_occupancy = 0.0
                    clear_observation_count = np.finfo(np.float32).max
                    clear_world_bbox = self.get_world_bounding_box().scale(2)
                    print("Override world bbox voxels")
                    self.get_mapper().perform_override_bounding_box_voxels(
                        clear_world_bbox, clear_occupancy, clear_observation_count, densify=False)
                    print("Load surface voxels")
                    self.get_mapper().perform_load_surface_voxels()
                    print("Storing octomap at key {}".format(cleared_world_bbox_with_surface_voxels_key))
                    self.get_mapper().store_octomap(cleared_world_bbox_with_surface_voxels_key)
                    print("Done")
                    reset_necessary = False

            location = min_location + self._prng.rand(3) * location_range
            assert(self.get_world_bounding_box().contains(location))
            pose = self.Pose(location, orientation_rpy)
            # Update pose so it is visible in Rviz
            self._update_pose(pose)

            if np.any(self._clear_extent > 0):
                clear_bbox = math_utils.BoundingBox(- self._clear_extent / 2.,
                                                    + self._clear_extent / 2.).move(location)
            else:
                clear_bbox = math_utils.BoundingBox.max_extent()

            valid_location = True
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    valid_location = False
                    # print("Sampled location with collision: {}".format(location))
                else:
                    for action_index in range(self.get_num_of_actions()):
                        pose_after_action = self.simulate_action_on_pose(pose, action_index)
                        if not clear_bbox.contains(pose_after_action.location()):
                            valid_location = False
                            # print("Action {} is colliding".format(action_index))
                            break
                        camera_bbox = self._collision_bbox.move(pose_after_action.location())
                        if not clear_bbox.contains(camera_bbox):
                            valid_location = False
                            # print("Camera bbox is not contained in clear bbox after action {}".format(action_index))
                            break
                        if self.is_action_colliding(pose, action_index):
                            # print("Action {} is colliding".format(action_index))
                            valid_location = False
                            break
                if valid_location and self._collision_check_above_camera:
                    if self.is_ray_colliding(pose.location(), pose.location() + np.array([0, 0, self._collision_check_above_camera_distance])):
                        # print("Sampled location with collision above")
                        valid_location = False
            if not valid_location:
                continue
            if valid_location:
                # Sanity check
                if np.any(self._clear_extent > 0):
                    res = self.get_mapper().perform_query_bbox(clear_bbox, 0)
                    occupancies = np.asarray(res.occupancies)
                    observation_certainties = np.asarray(res.observation_certainties)
                    if np.any(occupancies > self._collision_occupancy_threshold):
                        # print("WARNING: Invalid start location because of occupancies in clear bbox")
                        # print(occupancies)
                        valid_location = False
                    if np.any(observation_certainties < self._collision_observation_certainty_threshold):
                        # print("WARNING: Invalid start location because of observation certainties in clear bbox")
                        # print(observation_certainties)
                        valid_location = False
            if not valid_location:
                continue

            reset_necessary = True
            reset_result = super(Environment, self)._reset(pose, **kwargs)
            # Check if all actions are still possible after reset
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    valid_location = False
                    print("WARNING: Sampled location collides after map reset. Adjust the clearing bbox")
                for action_index in range(self.get_num_of_actions()):
                    if self.is_action_colliding(pose, action_index, verbose=True):
                        valid_location = False
                        print("WARNING: Sampled location collides for action {} after map reset. "
                              "Adjust the clearing bbox".format(action_index))
                        break
        self._after_reset_hooks()
        return reset_result

        # theta = self._get_origin_angle(location)
        # roll = 0
        # pitch = 0
        # if self._random_reset:
        #     # yaw = 2 * np.pi * self._prng.rand()
        #     u = 2 * (self._prng.rand() - 0.5)
        #     yaw = theta + np.pi + u * np.pi / 8.
        # else:
        #     yaw = 0
        # orientation_rpy = np.array([roll, pitch, yaw])
        # pose = self.Pose(location, orientation_rpy)

    # # TODO: Make proper collision detection
    def is_action_allowed_on_pose(self, pose, action_index):
        return True


class EnvironmentNoPitch(Environment):

    def __init__(self,
                 world_bounding_box,
                 random_reset=True,
                 move_distance=2.0,
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount_degrees=None,
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            move_distance (float): Scale of local motions.
            yaw_amount (float): Scale of yaw rotations.
            yaw_amount_degrees (float: Scale of yaw rotation in degrees. Overwrites yaw_amount if not `None`.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._move_distance = move_distance
        self._yaw_amount = yaw_amount
        if yaw_amount_degrees is not None:
            self._yaw_amount = math_utils.degrees_to_radians(yaw_amount_degrees)
        # self._pitch_amount = pitch_amount
        action_list = [
            # self.nop,
            self.move_left,
            self.move_right,
            self.move_down,
            self.move_up,
            self.move_backward,
            self.move_forward,
            self.yaw_clockwise,
            self.yaw_counter_clockwise,
            self.yaw_turn_around,
            # self.pitch_up,
            # self.pitch_down,
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
            # False,
            # False,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_list))
        super(Environment, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            **kwargs)

    def _get_origin_angle(self, location):
        theta = np.arctan2(location[1], location[0])
        return theta

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

    def yaw_turn_around(self, pose):
        """Perform yaw rotation around 180 degrees"""
        return self._rotate(pose, 0, 0, np.pi)

    #TODO: Fix random placement
    def reset(self, ignore_collision=False, pose=None, **kwargs):
        """Resets the environment."""
        override_pose = self._before_reset_hooks()
        if override_pose is not None:
            pose = override_pose
        if pose is not None:
            reset_result = super(Environment, self)._reset(pose, **kwargs)
            # Check if all actions are still possible after reset
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    print("WARNING: Sampled location collides after map reset. Adjust the clearing bbox")
                for action_index in range(self.get_num_of_actions()):
                    if self.is_action_colliding(pose, action_index, verbose=True):
                        print("WARNING: Sampled location collides for action {} after map reset. "
                              "Adjust the clearing bbox".format(action_index))
                        break
            self._after_reset_hooks()
            return reset_result
        roll = 0
        pitch = 0
        if self._random_reset:
            # yaw = 2 * np.pi * self._prng.rand()
            u = self._prng.rand()
            yaw = 2 * np.pi * u
        else:
            yaw = 0
        orientation_rpy = np.array([roll, pitch, yaw])
        center_location = 0.5 * (self.get_start_bounding_box().maximum() + self.get_start_bounding_box().minimum())
        location_range = self.get_start_bounding_box().maximum() - self.get_start_bounding_box().minimum()
        min_location = center_location - 0.5 * location_range
        valid_location = False
        reset_necessary = True
        while not valid_location:
            if reset_necessary:
                cleared_world_bbox_with_surface_voxels_key = "cleared_world_bbox_with_surface_voxels"
                if self.get_mapper().does_octomap_exist(cleared_world_bbox_with_surface_voxels_key):
                    print("Loading cleared octomap with surface voxels")
                    self.get_mapper().restore_octomap(cleared_world_bbox_with_surface_voxels_key)
                    reset_necessary = False
                else:
                    print("Map reset")
                    self.get_mapper().perform_reset()
                    clear_occupancy = 0.0
                    clear_observation_count = np.finfo(np.float32).max
                    clear_world_bbox = self.get_world_bounding_box().scale(2)
                    print("Override world bbox voxels")
                    self.get_mapper().perform_override_bounding_box_voxels(
                        clear_world_bbox, clear_occupancy, clear_observation_count, densify=False)
                    print("Load surface voxels")
                    self.get_mapper().perform_load_surface_voxels()
                    print("Storing octomap at key {}".format(cleared_world_bbox_with_surface_voxels_key))
                    self.get_mapper().store_octomap(cleared_world_bbox_with_surface_voxels_key)
                    print("Done")
                    reset_necessary = False

            location = min_location + self._prng.rand(3) * location_range
            assert(self.get_world_bounding_box().contains(location))
            pose = self.Pose(location, orientation_rpy)
            # Update pose so it is visible in Rviz
            self._update_pose(pose)

            if np.any(self._clear_extent > 0):
                clear_bbox = math_utils.BoundingBox(- self._clear_extent / 2.,
                                                    + self._clear_extent / 2.).move(location)
            else:
                clear_bbox = math_utils.BoundingBox.max_extent()

            valid_location = True
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    valid_location = False
                    # print("Sampled location with collision: {}".format(location))
                else:
                    for action_index in range(self.get_num_of_actions()):
                        pose_after_action = self.simulate_action_on_pose(pose, action_index)
                        if not clear_bbox.contains(pose_after_action.location()):
                            valid_location = False
                            # print("Action {} is colliding".format(action_index))
                            break
                        camera_bbox = self._collision_bbox.move(pose_after_action.location())
                        if not clear_bbox.contains(camera_bbox):
                            valid_location = False
                            # print("Camera bbox is not contained in clear bbox after action {}".format(action_index))
                            break
                        if self.is_action_colliding(pose, action_index):
                            # print("Action {} is colliding".format(action_index))
                            valid_location = False
                            break
                if valid_location and self._collision_check_above_camera:
                    if self.is_ray_colliding(pose.location(), pose.location() + np.array([0, 0, self._collision_check_above_camera_distance])):
                        # print("Sampled location with collision above")
                        valid_location = False
            if not valid_location:
                continue
            if valid_location:
                # Sanity check
                if np.any(self._clear_extent > 0):
                    res = self.get_mapper().perform_query_bbox(clear_bbox, 0)
                    occupancies = np.asarray(res.occupancies)
                    observation_certainties = np.asarray(res.observation_certainties)
                    if np.any(occupancies > self._collision_occupancy_threshold):
                        # print("WARNING: Invalid start location because of occupancies in clear bbox")
                        # print(occupancies)
                        valid_location = False
                    if np.any(observation_certainties < self._collision_observation_certainty_threshold):
                        # print("WARNING: Invalid start location because of observation certainties in clear bbox")
                        # print(observation_certainties)
                        valid_location = False
            if not valid_location:
                continue

            reset_necessary = True
            reset_result = super(Environment, self)._reset(pose, **kwargs)
            # Check if all actions are still possible after reset
            if not ignore_collision:
                if self.is_pose_colliding(pose, verbose=True):
                    valid_location = False
                    print("WARNING: Sampled location collides after map reset. Adjust the clearing bbox")
                for action_index in range(self.get_num_of_actions()):
                    if self.is_action_colliding(pose, action_index, verbose=True):
                        valid_location = False
                        print("WARNING: Sampled location collides for action {} after map reset. "
                              "Adjust the clearing bbox".format(action_index))
                        break
        self._after_reset_hooks()
        return reset_result


class SimpleV0Environment(BaseEnvironment):
    """SimpleV0Environment adds simple orbital motion actions to BaseEnvironment."""

    def __init__(self,
                 world_bounding_box,
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 action_not_allowed_reward=DEFAULT_ACTION_NOT_ALLOWED_REWARD,
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        action_list = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise,
        ]
        update_map_flags = [
            True,
            True,
        ]
        action_penalty = DEFAULT_ACTION_PENALTY
        action_rewards = np.array([action_penalty] * len(action_list))
        super(SimpleV0Environment, self).__init__(
            world_bounding_box,
            action_list,
            update_map_flags=update_map_flags,
            action_rewards=action_rewards,
            **kwargs)

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
            theta = 2 * np.pi * self._prng.rand()
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
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        action_list = [
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
            action_list,
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
            theta = 2 * np.pi * self._prng.rand()
            if self._prng.rand() < 0.25:
                yaw = 2 * np.pi * self._prng.rand()
            else:
                d_yaw = np.pi / 4 * (self._prng.rand() - 0.5)
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
                 random_reset=True,
                 radius=7.0,
                 height=3.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 8.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 8.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        action_list = [
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
            action_list,
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
            theta = 2 * np.pi * self._prng.rand()
            if self._prng.rand() < 0.25:
                yaw = 2 * np.pi * self._prng.rand()
            else:
                d_yaw = np.pi / 4 * (self._prng.rand() - 0.5)
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
                 random_reset=True,
                 radius=15.0,
                 height=5.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 yaw_amount=math_utils.degrees_to_radians(180. / 10.),
                 pitch_amount=math_utils.degrees_to_radians(180. / 5.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            yaw_amount (float): Scale of yaw rotations.
            pitch_amount (float): Scale of pitch rotations.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        self._yaw_amount = yaw_amount
        self._pitch_amount = pitch_amount
        action_list = [
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
        action_rewards = np.array([action_penalty] * len(action_list))
        super(SimpleV3Environment, self).__init__(
            world_bounding_box,
            action_list,
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
            theta = 2 * np.pi * self._prng.rand()
            yaw = 2 * np.pi * self._prng.rand()
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
                 random_reset=True,
                 radius=15.0,
                 height=5.0,
                 angle_amount=math_utils.degrees_to_radians(180. / 10.),
                 **kwargs):
        """Initialize environment.

        Args:
            world_bounding_box (BoundingBox): Overall bounding box of the world to restrict motion.
            engine (BaseEngine): Simulation engine (i.e. Unreal Engine wrapper).
            mapper: Occupancy mapper (i.e. OctomapExt interface).
            clear_extent (float): Size of bounding box to clear in the occupancy map on reset.
            random_reset (bool): Use random pose when resetting.
            radius (float): Radius of orbit.
            height (float): Height of orbit.
            angle_amount (float): Scale of orbital motion.
            action_not_allowed_reward (float): Reward value for invalid actions (i.e. collision).
        """

        self._random_reset = random_reset
        self._radius = radius
        self._height = height
        self._angle_amount = angle_amount
        action_list = [
            self.orbit_clockwise,
            self.orbit_counter_clockwise
        ]
        super(VerySimpleEnvironment, self).__init__(
            world_bounding_box,
            action_list,
            **kwargs)

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
            theta = 2 * np.pi * self._prng.rand()
        else:
            theta = 0
        pose = self._get_orbit_pose(theta)
        super(VerySimpleEnvironment, self).reset(pose, **kwargs)

    def is_action_allowed_on_pose(self, pose, action_index):
        return True
