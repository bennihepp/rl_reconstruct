import time
import numpy as np
import rospy
from tf import transformations
from RLrecon import math_utils


class BaseEngine(object):

    @staticmethod
    def _create_points_from_depth_image(depth_image, focal_length):
        """Returns point cloud in camera frame. Assuming x-axis points forward, y-axis left and z-axis up."""
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

    def get_depth_point_cloud_rpy(self, location, orientation_rpy):
        self.set_location(location)
        roll, pitch, yaw = orientation_rpy
        self.set_orientation_rpy(roll, pitch, yaw)
        focal_length = self.get_focal_length()
        # TODO: Seems to be unnecessary now.
        # Make sure that the new frame is rendered.
        # This should be fixed in the UnrealCV plugin.
        quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        max_delay_time = 2.0
        t0 = time.time()
        while time.time() - t0 < max_delay_time:
            if math_utils.is_vector_equal(location, self.get_location(), tolerance=1e-4) \
                    and math_utils.is_equal_quaternion(quat, self.get_orientation_quat(), tolerance=1e-4):
                break
            time.sleep(0.01)
        # TODO: Seems to be unnecessary now.
        # After the pose was successfully updated, wait a little bit longer.
        # time.sleep(0.1)
        # t1 = time.time()
        # print("delay={}".format(t1 - t0))
        depth_image = self.get_depth_image()
        rospy.logdebug("min depth={}, max depth={}".format(
            np.min(depth_image.flatten()), np.max(depth_image.flatten())))

        # Create pointcloud message
        rospy.logdebug("Creating point cloud message")
        points = self._create_points_from_depth_image(depth_image, focal_length)
        return points

    def get_depth_point_cloud_world_rpy(self, location, orientation_rpy):
        points = self.get_depth_point_cloud_rpy(location, orientation_rpy)
        world_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        world_trans = location
        world_rot_mat = transformations.quaternion_matrix(world_quat)[:3, :3]
        points_world = points.dot(world_rot_mat.T) + world_trans
        return points_world
