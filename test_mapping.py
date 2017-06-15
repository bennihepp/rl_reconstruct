import sys
import struct
import numpy as np
import rospy
import tf
from tf import transformations
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from engine.unreal_cv_wrapper import UnrealCVWrapper


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
    #     # print('{} {} {}'.format(*[points[i][j] for j in xrange(3)]))
    # # for point in points:
    # #     points_flat.append(float(point[0]))
    # #     points_flat.append(float(point[1]))
    # #     points_flat.append(float(point[2]))
    point_cloud_msg.data = struct.pack("={}f".format(len(points_flat)), *points_flat)
    return point_cloud_msg


def run(engine, point_cloud_publisher, location_origin):
    tf_br = tf.TransformBroadcaster()
    world_frame = "map"
    sensor_frame = "depth_sensor"
    rate = rospy.Rate(5)
    yaw = 0
    while not rospy.is_shutdown():
        # Read new pose, camera info and depth image
        rospy.loginfo("Reading pose")
        pose = engine.get_pose()
        focal_length = engine.get_focal_length()
        depth_image = engine.get_depth_image()
        rospy.loginfo("min depth={}, max depth={}".format(
            np.min(depth_image.flatten()), np.max(depth_image.flatten())))
        # Publish transform
        location = pose[0] - location_origin
        quaternion = pose[1]
        euler_ypr = np.array(transformations.euler_from_quaternion(quaternion, 'rzyx'))
        euler_rpy = engine.get_orientation_rpy()
        print('location: {}'.format(location))
        print('quaternion: {}'.format(quaternion))
        print('euler_ypr: {}'.format(euler_ypr * 180 / np.pi))
        print('euler_rpy: {}'.format(euler_rpy * 180 / np.pi))
        timestamp = rospy.Time.now()
        tf_br.sendTransform(
            location, quaternion,
            timestamp,
            sensor_frame, world_frame)
        # Create pointcloud and publish
        rospy.loginfo("Creating point cloud message")
        points = create_points_from_depth_image(pose, depth_image, focal_length)
        # points = create_points_synthetic()
        point_cloud_msg = create_point_cloud_msg(points)
        point_cloud_msg.header.stamp = timestamp
        rospy.loginfo("Publishing point cloud")
        point_cloud_publisher.publish(point_cloud_msg)
        # Perform action
        # Update camera orientation
        roll = 0
        pitch = 0
        yaw += 25 * np.pi / 180.
        if yaw >= 2 * np.pi:
            yaw -= 2 * np.pi
        elif yaw < 0:
            yaw += 2 * np.pi
        engine.set_orientation_rpy(roll, pitch, yaw)

        rate.sleep()


if __name__ == '__main__':
    engine = UnrealCVWrapper()
    point_cloud_pub = rospy.Publisher('cloud_in', PointCloud2, queue_size=1)
    location_origin = np.array([0, 0, 0])
    rospy.init_node('test_mapping', anonymous=False)
    run(engine, point_cloud_pub, location_origin)
