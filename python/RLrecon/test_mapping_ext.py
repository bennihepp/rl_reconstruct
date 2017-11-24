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
from octomap_server_ext.srv import InsertPointCloud, InsertPointCloudRequest
from RLrecon.contrib import transformations
from RLrecon.engines.unreal_cv_wrapper_old import UnrealCVWrapper


"""Returns point cloud in camera frame. Assuming x-axis points forward, y-axis left and z-axis up."""
def create_points_from_depth_image(pose, depth_image, focal_length):
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    x_c = np.float(width) / 2 - 1
    y_c = np.float(height) / 2 - 1
    points = np.empty((depth_image.shape[0] * depth_image.shape[1], 3))
    points[:, -1] = depth_image.flatten()
    columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
    points[:, 0] = points[:, -1] * (columns.flatten() - x_c) / focal_length
    points[:, 1] = points[:, -1] * (rows.flatten() - y_c) / focal_length
    # Transform points into camera coordinate system (x-axis forward, z-axis up, y-axis left)
    quat1 = transformations.quaternion_about_axis(-np.pi / 2, [1, 0, 0])
    quat2 = transformations.quaternion_about_axis(-np.pi / 2, [0, 0, 1])
    quat = transformations.quaternion_multiply(quat2, quat1)
    rot_mat = transformations.quaternion_matrix(quat)[:3, :3]
    points_transformed = points.dot(rot_mat.T)
    return points_transformed


def create_points_synthetic():
    points = []
    n1 = 10
    n2 = 10
    points = np.empty((n1 * n2, 3))
    for i in xrange(10):
        for j in xrange(10):
            x = 1
            y = 1 + j / 5.
            z = 1 + i / 10.
            # points.append(np.array([x, y, z]))
            points[i * n1 + j, :] = np.array([x, y, z])
    return points


def create_point_cloud_msg(points):
    point_cloud_msg = PointCloud2()
    point_cloud_msg.header.frame_id = 'depth_sensor'
    point_cloud_msg.height = 1
    # point_cloud_msg.width = len(points)
    point_cloud_msg.width = points.shape[0]
    point_cloud_msg.fields = []
    for i, name in enumerate(['x', 'y', 'z']):
        field = PointField()
        field.name = name
        field.offset = i * struct.calcsize('f')
        field.datatype = 7
        field.count = 1
        point_cloud_msg.fields.append(field)
    point_cloud_msg.is_bigendian = sys.byteorder == 'little'
    point_cloud_msg.point_step = 3 * struct.calcsize('f')
    point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
    point_cloud_msg.is_dense = True
    points_flat = points.flatten()
    # points_flat = np.empty(len(points) * 3, dtype=np.float)
    # for i in xrange(len(points)):
    #     points_flat[3 * i + 0] = float(points[i][0])
    #     points_flat[3 * i + 1] = float(points[i][1])
    #     points_flat[3 * i + 2] = float(points[i][2])
    #     # rospy.logdebug('{} {} {}'.format(*[points[i][j] for j in xrange(3)]))
    # # for point in points:
    # #     points_flat.append(float(point[0]))
    # #     points_flat.append(float(point[1]))
    # #     points_flat.append(float(point[2]))
    point_cloud_msg.data = struct.pack("={}f".format(len(points_flat)), *points_flat)
    return point_cloud_msg


def run(engine, point_cloud_topic, location_origin):
    rospy.wait_for_service(point_cloud_topic)
    point_cloud_service = rospy.ServiceProxy(point_cloud_topic, InsertPointCloud, persistent=True)

    # Setup loop timer
    rate = rospy.Rate(0.5)

    # center_location = engine.get_location()
    center_location = np.array([0, 0, 2])
    rospy.loginfo("Center location: {}".format(center_location))
    radius = 7.5
    look_inside = True

    # Initial angle
    yaw = 0

    # Image size scale factor
    scale_factor = 0.5

    while not rospy.is_shutdown():
        # Perform action:
        # Update camera orientation
        roll = 0
        pitch = 0
        if yaw >= 2 * np.pi:
            yaw -= 2 * np.pi
        elif yaw < 0:
            yaw += 2 * np.pi
        engine.set_orientation_rpy(roll, pitch, yaw)
        # Update camera location
        new_location = np.copy(center_location)
        if look_inside:
            new_location[0] += radius * np.cos(yaw + np.pi)
            new_location[1] += radius * np.sin(yaw + np.pi)
        else:
            new_location[0] += radius * np.cos(yaw)
            new_location[1] += radius * np.sin(yaw)
        engine.set_location(new_location)
        rospy.logdebug("location: {} {} {}".format(*new_location))
        rospy.logdebug("rotation: {} {} {}".format(roll * 180 / np.pi, pitch * 180 / np.pi, yaw * 180 / np.pi))
        # Update camera state
        yaw += 25 * np.pi / 180.

        # Read new pose, camera info and depth image
        rospy.logdebug("Reading pose")
        pose = engine.get_pose()
        focal_length = engine.get_focal_length() * scale_factor
        depth_image = engine.get_depth_image()
        dsize = (int(depth_image.shape[1] * scale_factor), int(depth_image.shape[0] * scale_factor))
        depth_image = cv2.resize(depth_image, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        rospy.logdebug("min depth={}, max depth={}".format(
            np.min(depth_image.flatten()), np.max(depth_image.flatten())))

        # Publish transform
        location = pose[0] - location_origin
        quaternion = pose[1]
        euler_ypr = np.array(transformations.euler_from_quaternion(quaternion, 'rzyx'))
        euler_rpy = engine.get_orientation_rpy()
        # rospy.logdebug('location: {}'.format(location))
        # rospy.logdebug('quaternion: {}'.format(quaternion))
        # rospy.logdebug('euler_ypr: {}'.format(euler_ypr * 180 / np.pi))
        # rospy.logdebug('euler_rpy: {}'.format(euler_rpy * 180 / np.pi))
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

        # Create pointcloud message
        rospy.logdebug("Creating point cloud message")
        points = create_points_from_depth_image(pose, depth_image, focal_length)
        # points = create_points_synthetic()
        point_cloud_msg = create_point_cloud_msg(points)
        point_cloud_msg.header.stamp = timestamp

        # Request point cloud insertion
        rospy.logdebug("Requesting point cloud insertion")
        try:
            request = InsertPointCloudRequest()
            request.point_cloud = point_cloud_msg
            request.sensor_to_world = sensor_to_world
            response = point_cloud_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Point cloud service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            point_cloud_service = rospy.ServiceProxy(point_cloud_topic, InsertPointCloud, persistent=True)
        else:
            rospy.loginfo("Integrating point cloud took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Received score: {}".format(response.score))
            rospy.loginfo("Received reward: {}".format(response.reward))

        rate.sleep()


if __name__ == '__main__':
    engine = UnrealCVWrapper(image_scale_factor=1.0)
    point_cloud_topic = '/octomap_server_ext/insert_point_cloud'
    location_origin = np.array([0, 0, 0])
    rospy.init_node('test_mapping', anonymous=False)
    run(engine, point_cloud_topic, location_origin)
