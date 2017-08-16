import sys
import struct
import numpy as np
import cv2
import rospy
import tf
import tf_conversions
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from octomap_server_ext.srv import RaycastCamera, RaycastCameraRequest
from RLrecon.contrib import transformations
from RLrecon.engines.unreal_cv_wrapper import UnrealCVWrapper
import math_utils
import ros_utils


def get_bbox_mask(point_cloud, bbox):
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


def get_uncertain_mask(point_cloud):
    uncertain_voxel_mask = \
        np.logical_or(
            np.logical_not(point_cloud['is_known']),
            np.logical_and(
                point_cloud['occupancy'] >= 0.25,
                point_cloud['occupancy'] <= 0.75
            )
        )
    return uncertain_voxel_mask


def run(engine, raycast_topic, location_origin):
    rospy.wait_for_service(raycast_topic)
    raycast_service = rospy.ServiceProxy(raycast_topic, RaycastCamera, persistent=True)
    raycast_pc_pub = rospy.Publisher("raycast_point_cloud", PointCloud2, queue_size=1)
    uncertain_pc_pub = rospy.Publisher("uncertain_point_cloud", PointCloud2, queue_size=1)

    # Setup loop timer
    rate = rospy.Rate(10)

    # Raycast camera model
    height = 240
    width = 320
    focal_length = 160
    ignore_unknown_voxels = False

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
            bounding_box = math_utils.BoundingBox(
                [-3, -3, 0],
                [+3, +3, +6]
            )
            pc = ros_utils.point_cloud2_ros_to_numpy(response.point_cloud)
            pc = np.unique(pc)
            bbox_mask = get_bbox_mask(pc, bounding_box)
            pc = pc[bbox_mask]
            print(pc.shape)
            uncertain_mask = get_uncertain_mask(pc)
            pc = pc[uncertain_mask]
            tentative_reward = len(pc)
            rospy.loginfo("Tentative reward: {}".format(tentative_reward))
            rospy.loginfo("")
            raycast_pc_msg = response.point_cloud
            raycast_pc_msg.header.stamp = timestamp
            raycast_pc_msg.header.frame_id = 'map'
            raycast_pc_pub.publish(raycast_pc_msg)
            pc_xyz = ros_utils.structured_to_3d_array(pc)
            print("pc_xyz.shape: ", pc_xyz.shape)
            uncertain_pc_msg = ros_utils.point_cloud2_numpy_to_ros(pc_xyz)
            uncertain_pc_msg.header.stamp = timestamp
            uncertain_pc_msg.header.frame_id = 'map'
            uncertain_pc_pub.publish(uncertain_pc_msg)

        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('test_mapping', anonymous=False)
    engine = UnrealCVWrapper()
    raycast_topic = 'raycast_camera'
    location_origin = np.array([0, 0, 0])
    run(engine, raycast_topic, location_origin)
