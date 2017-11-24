import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from pybh import ros_utils, math_utils


class RosEnvironmentHooks(object):

    def __init__(self, environment, pose_topic='agent_pose', pose_trace_topic='agent_pose_trace',
                 actions_topic='agent_actions',
                 world_frame='map', node_namespace="/environment/"):
        """Initialize base environment.

        Args:
            environment (BaseEnvironment): Environment for which to register hooks.
            pose_topic (str): Topic for agent pose.
            world_frame (str): Id of the world frame.
            node_namespace (str): Ros namespace for this node.
        """
        import rospy
        from geometry_msgs.msg import PoseStamped
        self._environment = environment
        self._world_frame = world_frame
        self._pose_pub = rospy.Publisher(node_namespace + pose_topic, PoseStamped, queue_size=10)
        self._pose_trace_pub = rospy.Publisher(node_namespace + pose_trace_topic, MarkerArray, queue_size=10)
        self._actions_pub = rospy.Publisher(node_namespace + actions_topic, MarkerArray, queue_size=10)
        self._pose_trace = []
        environment.after_reset_hooks.register(self.after_reset_cb)
        environment.update_pose_hooks.register(self.update_pose_cb)

    def after_reset_cb(self):
        # Delete old pose trace markers
        # marker_arr_msg = self._create_pose_trace_marker_array(self._pose_trace, delete=True)
        print("Clearing agent pose trace")
        marker_arr_msg = self._create_dummy_pose_trace_marker_array(1000, delete=True)
        self._pose_trace_pub.publish(marker_arr_msg)
        self._pose_trace = []
        marker_arr_msg = self._create_dummy_actions_marker_array(100, delete=True)
        self._actions_pub.publish(marker_arr_msg)

    def update_pose_cb(self, pose):
        self._publish_actions(pose)
        self._publish_pose(pose)
        self._pose_trace.append(pose)
        self._publish_pose_trace(self._pose_trace)

    def _pose_to_ros_pose_msg(self, pose, time=None):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self._world_frame
        if time is None:
            pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position = ros_utils.point_numpy_to_ros(pose.location())
        orientation_quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
        pose_msg.pose.orientation = ros_utils.quaternion_numpy_to_ros(orientation_quat)
        return pose_msg

    def _publish_pose(self, pose):
        """Publish pose if ROS is enabled"""
        pose_msg = self._pose_to_ros_pose_msg(pose)
        self._pose_pub.publish(pose_msg)

    def _fill_ros_marker_msg_with_pose(self, marker_msg, pose):
        marker_msg.pose.position = ros_utils.point_numpy_to_ros(pose.location())
        orientation_quat = math_utils.convert_rpy_to_quat(pose.orientation_rpy())
        marker_msg.pose.orientation = ros_utils.quaternion_numpy_to_ros(orientation_quat)

    def _create_pose_trace_marker_array(self, pose_trace, delete=False):
        marker_arr_msg = MarkerArray()
        for i, pose in enumerate(pose_trace):
            marker_msg = Marker()
            marker_msg.header.frame_id = self._world_frame
            marker_msg.header.stamp = rospy.Time.now()
            marker_msg.ns = "pose_trace"
            marker_msg.id = i
            marker_msg.type = Marker.ARROW
            if delete:
                marker_msg.action = Marker.DELETE
            else:
                marker_msg.action = Marker.ADD
            self._fill_ros_marker_msg_with_pose(marker_msg, pose)
            marker_msg.scale.x = 1.0
            marker_msg.scale.y = 0.2
            marker_msg.scale.z = 0.2
            marker_msg.color.r = 1.0 * (len(pose_trace) - i) / float(len(pose_trace))
            marker_msg.color.g = 1.0
            # marker_msg.color.b = 1.0 * i / float(len(pose_trace))
            marker_msg.color.b = 0.0
            marker_msg.color.a = 1.0
            marker_arr_msg.markers.append(marker_msg)
        return marker_arr_msg

    def _create_dummy_pose_trace_marker_array(self, num_markers, delete=False):
        marker_arr_msg = MarkerArray()
        for i in range(num_markers):
            marker_msg = Marker()
            marker_msg.header.frame_id = self._world_frame
            marker_msg.header.stamp = rospy.Time.now()
            marker_msg.ns = "pose_trace"
            marker_msg.id = i
            marker_msg.type = Marker.ARROW
            if delete:
                marker_msg.action = Marker.DELETE
            else:
                marker_msg.action = Marker.ADD
            marker_arr_msg.markers.append(marker_msg)
        return marker_arr_msg

    def _publish_pose_trace(self, pose_trace):
        """Publish pose as marker trace"""
        marker_arr_msg = self._create_pose_trace_marker_array(pose_trace)
        self._pose_trace_pub.publish(marker_arr_msg)

    def _create_actions_marker_array(self, pose, delete=False):
        marker_arr_msg = MarkerArray()
        ros_time = rospy.Time.now()
        for action_index in range(self._environment.get_num_of_actions()):
            new_pose = self._environment.simulate_action_on_pose(pose, action_index)
            collision = self._environment.is_action_colliding(pose, action_index)
            marker_msg = Marker()
            marker_msg.header.frame_id = self._world_frame
            marker_msg.header.stamp = ros_time
            marker_msg.ns = "action"
            marker_msg.id = action_index
            marker_msg.type = Marker.ARROW
            if delete:
                marker_msg.action = Marker.DELETE
            else:
                marker_msg.action = Marker.ADD
            self._fill_ros_marker_msg_with_pose(marker_msg, new_pose)
            marker_msg.scale.x = 1.0
            marker_msg.scale.y = 0.2
            marker_msg.scale.z = 0.2
            if collision:
                marker_msg.color.r = 1.0
                marker_msg.color.g = 0.0
                marker_msg.color.b = 0.0
                marker_msg.color.a = 0.7
            else:
                marker_msg.color.r = 0.0
                marker_msg.color.g = 1.0
                marker_msg.color.b = 0.0
                marker_msg.color.a = 0.7
            marker_arr_msg.markers.append(marker_msg)
        return marker_arr_msg

    def _create_dummy_actions_marker_array(self, num_actions, delete=False):
        marker_arr_msg = MarkerArray()
        ros_time = rospy.Time.now()
        for action_index in range(num_actions):
            marker_msg = Marker()
            marker_msg.header.frame_id = self._world_frame
            marker_msg.header.stamp = ros_time
            marker_msg.ns = "action"
            marker_msg.id = action_index
            marker_msg.type = Marker.ARROW
            if delete:
                marker_msg.action = Marker.DELETE
            else:
                marker_msg.action = Marker.ADD
            marker_arr_msg.markers.append(marker_msg)
        return marker_arr_msg

    def _publish_actions(self, pose):
        """Publish possible actions and color them based on collisions if ROS is enabled"""
        marker_arr_msg = self._create_actions_marker_array(pose)
        self._actions_pub.publish(marker_arr_msg)
