import numpy as np
import math_utils


class Environment(object):

    class State(object):

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

    def __init__(self,
                 bounding_box,
                 engine=None,
                 mapper=None,
                 move_distance=2.0,
                 yaw_amount=np.pi / 10.,
                 pitch_amount=np.pi / 5.,
                 ignore_unknown_voxels=False,
                 action_not_allowed_reward=-100.,
                 use_ros=True,
                 ros_pose_topic='agent_pose',
                 ros_world_frame='map'):
        self._bounding_box = bounding_box
        if engine is None:
            from engine.unreal_cv_wrapper import UnrealCVWrapper
            engine = UnrealCVWrapper()
        self._engine = engine
        if mapper is None:
            from mapping.octomap_ext_mapper import OctomapExtMapper
            mapper = OctomapExtMapper()
        self._mapper = mapper
        self._action_map = [
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
        self._move_distance = move_distance
        self._yaw_amount = yaw_amount
        self._pitch_amount = pitch_amount
        self._ignore_unknown_voxels = ignore_unknown_voxels
        self._action_not_allowed_reward = action_not_allowed_reward
        if use_ros:
            import rospy
            from geometry_msgs.msg import PoseStamped
            self._use_ros = True
            self._ros_world_frame = ros_world_frame
            self._pose_pub = rospy.Publisher(ros_pose_topic, PoseStamped, queue_size=10)
        else:
            self._use_ros = False
        self._is_running = True

    def is_running(self):
        return self._is_running

    def stop(self):
        self._is_running = False

    def start(self):
        self._is_running = True

    def get_bounding_box(self):
        return self._bounding_box

    def get_state(self):
        location = self._engine.get_location()
        orientation_rpy = self._engine.get_orientation_rpy()
        state = self.State(location, orientation_rpy)
        self._publish_pose(state)
        return state

    def _publish_pose(self, state):
        if self._use_ros:
            import rospy
            from geometry_msgs.msg import PoseStamped
            import ros_utils
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self._ros_world_frame
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position = ros_utils.point_numpy_to_ros(state.location())
            orientation_quat = math_utils.convert_rpy_to_quat(state.orientation_rpy())
            pose_msg.pose.orientation = ros_utils.quaternion_numpy_to_ros(orientation_quat)
            self._pose_pub.publish(pose_msg)

    def reset_state(self, state):
        self._update_state(state)

    def is_action_allowed(self, state, action_index):
        location = state.location()
        new_state = self.simulate_action(state, action_index)
        new_location = new_state.location()
        if np.any(new_location < self._bounding_box.minimum()):
            return False
        if np.any(new_location > self._bounding_box.maximum()):
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

    def _update_state(self, new_state):
        self._engine.set_location(new_state.location())
        roll, pitch, yaw = new_state.orientation_rpy()
        self._engine.set_orientation_rpy(roll, pitch, yaw)
        self._publish_pose(new_state)

    def _get_depth_point_cloud(self, state):
        point_cloud = self._engine.get_depth_point_cloud_rpy(state.location(), state.orientation_rpy())
        return point_cloud

    def _get_depth_point_cloud_world(self, state):
        point_cloud = self._engine.get_depth_point_cloud_world_rpy(state.location(), state.orientation_rpy())
        return point_cloud

    def _update_map(self, state):
        point_cloud = self._get_depth_point_cloud(state)
        result = self._mapper.update_map_rpy(state.location(), state.orientation_rpy(), point_cloud)
        return result.reward

    def simulate_action(self, state, action_index):
        new_state = self._action_map[action_index](state)
        return new_state

    def initialize(self, clear_size=3, current_state=None):
        if current_state is None:
            current_state = self.get_state()
        else:
            self._update_state(current_state)
        clear_bbox = math_utils.BoundingBox(
            current_state.location() - clear_size,
            current_state.location() + clear_size)
        self._mapper.perform_clear_bounding_box_voxels(clear_bbox)
        # self._mapper.perform_override_bounding_box_voxels(clear_bbox, 0.7)

    def num_actions(self):
        return len(self._action_map)

    def perform_action(self, action_index, current_state=None):
        if current_state is None:
            current_state = self.get_state()
        else:
            self._update_state(current_state)
        if not self.is_action_allowed(current_state, action_index):
            return current_state, self._action_not_allowed_reward
        new_state = self._action_map[action_index](current_state)
        self._update_state(new_state)
        reward = self._update_map(new_state)
        return new_state, reward

    def get_action_name(self, action_index):
        return self._action_map[action_index].__name__

    def _move(self, state, offset):
        new_location = state.location() + offset
        return self.State(new_location, state.orientation_rpy())

    def _move_local(self, state, local_offset):
        quat = math_utils.convert_rpy_to_quat(state.orientation_rpy())
        world_offset = math_utils.rotate_vector_with_quaternion(quat, local_offset)
        return self._move(state, world_offset[:3])

    def _rotate(self, state, d_roll, d_pitch, d_yaw):
        new_rpy = state.orientation_rpy() + np.array([d_roll, d_pitch, d_yaw])
        # print("New rpy: {}".format(new_rpy))
        if new_rpy[1] > np.pi / 2:
            new_rpy[1] = np.pi / 2
        elif new_rpy[1] < 0:
            new_rpy[1] = 0
        # print("Corrected new rpy: {}".format(new_rpy))
        return self.State(state.location(), new_rpy)

    def nop(self, state):
        return self.State(state.location(), state.orientation_rpy())

    def move_left(self, state):
        return self._move_local(state, np.array([0, +self._move_distance, 0]))

    def move_right(self, state):
        return self._move_local(state, np.array([0, -self._move_distance, 0]))

    def move_down(self, state):
        return self._move_local(state, np.array([0, 0, -self._move_distance]))

    def move_up(self, state):
        return self._move_local(state, np.array([0, 0, +self._move_distance]))

    def move_backward(self, state):
        return self._move_local(state, np.array([-self._move_distance, 0, 0]))

    def move_forward(self, state):
        return self._move_local(state, np.array([+self._move_distance, 0, 0]))

    def yaw_clockwise(self, state):
        return self._rotate(state, 0, 0, -self._yaw_amount)

    def yaw_counter_clockwise(self, state):
        return self._rotate(state, 0, 0, +self._yaw_amount)

    def pitch_up(self, state):
        return self._rotate(state, 0, -self._pitch_amount, 0)

    def pitch_down(self, state):
        return self._rotate(state, 0, +self._pitch_amount, 0)
