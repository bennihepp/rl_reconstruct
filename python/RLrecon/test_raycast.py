import sys
import struct
import numpy as np
import cv2
import rospy
import tf
from tf import transformations
import tf_conversions
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from octomap_server_ext.srv import RaycastCamera, RaycastCameraRequest
from RLrecon.engine.unreal_cv_wrapper import UnrealCVWrapper


def run(engine, raycast_topic, location_origin):
    rospy.wait_for_service(raycast_topic)
    raycast_service = rospy.ServiceProxy(raycast_topic, RaycastCamera, persistent=True)
    raycast_pc_pub = rospy.Publisher("raycast_point_cloud", PointCloud2, queue_size=1)

    # Setup loop timer
    rate = rospy.Rate(10)

    # Raycast camera model
    height = 240
    width = 320
    focal_length = 160
    ignore_unknown_voxels = True

    while not rospy.is_shutdown():
        # Read new pose, camera info and depth image
        rospy.logdebug("Reading pose")
        pose = engine.get_pose()
        # Publish transform
        location = pose[0] - location_origin
        quaternion = pose[1]
        euler_ypr = np.array(transformations.euler_from_quaternion(quaternion, 'rzyx'))
        euler_rpy = engine.get_orientation_rpy()
        # rospy.logdebug('location: {}'.format(location))
        # rospy.logdebug('quaternion: {}'.format(quaternion))
        # rospy.logdebug('euler_ypr: {}'.format(euler_ypr * 180 / np.pi))
        # rospy.loginfo('euler_rpy: {}'.format(euler_rpy * 180 / np.pi))
        timestamp = rospy.Time.now()

        # Create transform message
        transform_mat = transformations.quaternion_matrix(quaternion)
        transform_mat[:3, 3] = location
        quat = transformations.quaternion_from_matrix(transform_mat)
        trans = transformations.translation_from_matrix(transform_mat)
        sensor_to_world = Transform()
        sensor_to_world.translation.x = location[0]
        sensor_to_world.translation.y = location[1]
        sensor_to_world.translation.z = location[2]
        sensor_to_world.rotation.x = quaternion[0]
        sensor_to_world.rotation.y = quaternion[1]
        sensor_to_world.rotation.z = quaternion[2]
        sensor_to_world.rotation.w = quaternion[3]

        # Request raycast
        rospy.logdebug("Requesting raycast")
        try:
            request = RaycastCameraRequest()
            request.sensor_to_world = sensor_to_world
            request.height = height
            request.width = width
            request.focal_length = focal_length
            request.ignore_unknown_voxels = ignore_unknown_voxels
            response = raycast_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Raycast service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            raycast_service = rospy.ServiceProxy(raycast_topic, RaycastCamera, persistent=True)
        else:
            rospy.loginfo("Raycast took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Number of hit occupied voxels: {}".format(response.num_hits_occupied))
            rospy.loginfo("Number of hit unknown voxels: {}".format(response.num_hits_unknown))
            rospy.loginfo("Expected reward: {}".format(response.expected_reward))
            rospy.loginfo("Point cloud size: {}".format(response.point_cloud.width * response.point_cloud.height))
            pc = response.point_cloud
            pc.header.stamp = timestamp
            pc.header.frame_id = 'map'
            raycast_pc_pub.publish(pc)

        rate.sleep()


if __name__ == '__main__':
    engine = UnrealCVWrapper()
    raycast_topic = 'raycast_camera'
    location_origin = np.array([0, 0, 0])
    rospy.init_node('test_mapping', anonymous=False)
    run(engine, raycast_topic, location_origin)
