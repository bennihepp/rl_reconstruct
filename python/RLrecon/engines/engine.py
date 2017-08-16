from __future__ import print_function
import time
import numpy as np
import rospy
from RLrecon.contrib import transformations
from RLrecon import math_utils
from RLrecon.utils import Timer


class BaseEngine(object):

    def __init__(self,
                 max_depth_distance=np.finfo(np.float).max,
                 max_depth_viewing_angle=math_utils.degrees_to_radians(90.)):
        self._max_depth_distance = max_depth_distance
        self._max_depth_viewing_dot_prod = np.cos(max_depth_viewing_angle)
        self._ray_direction_image = None

    def _compute_normal_image_dot_product(self, normal_image, other_vec):
        dot_prod = np.tensordot(normal_image, other_vec, axes=1)
        return dot_prod

    def filter_normal_image(self, normal_image, ray_directions=None, inplace=True):
        """Filter normal image based on maximum viewing angle"""
        if ray_directions is None:
            ray_directions = self.get_ray_direction_image_world(
                normal_image.shape[1], normal_image.shape[0], self.get_focal_length())
        # dot_prod = self._compute_normal_image_dot_product(normal_image, -view_direction)
        dot_prod = np.sum(normal_image * (-ray_directions), axis=2)
        if inplace:
            filtered_normal_image = normal_image
        else:
            filtered_normal_image = np.array(normal_image)
        filtered_normal_image[dot_prod < self._max_depth_viewing_dot_prod, :] = 0
        return filtered_normal_image

    def filter_depth_image(self, depth_image, normal_image, ray_directions=None, inplace=True):
        """Filter depth image based on maximum viewing angle"""
        if ray_directions is None:
            ray_directions = self.get_ray_direction_image_world(
                depth_image.shape[1], depth_image.shape[0], self.get_focal_length())
        # dot_prod = self._compute_normal_image_dot_product(normal_image, -view_direction)
        dot_prod = np.sum(normal_image * (-ray_directions), axis=2)
        if inplace:
            filtered_depth_image = depth_image
        else:
            filtered_depth_image = np.array(depth_image)
        filtered_depth_image[depth_image > self._max_depth_distance] = -1.0
        filtered_depth_image[dot_prod < self._max_depth_viewing_dot_prod] = -1.0
        return filtered_depth_image

    def _create_points_from_depth_image(self, depth_image, focal_length):
        """Return point cloud in camera frame. Assuming x-axis points forward, y-axis left and z-axis up."""
        # timer = Timer()
        height = depth_image.shape[0]
        width = depth_image.shape[1]
        x_c = np.float(width) / 2 - 1
        y_c = np.float(height) / 2 - 1
        depth_image_flat = depth_image.flatten()
        points = np.empty((depth_image_flat.shape[0], 3))
        # t1 = timer.elapsed_seconds()
        points[:, -1] = depth_image_flat
        # t2 = timer.elapsed_seconds()
        columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
        # t3 = timer.elapsed_seconds()
        points[:, 0] = points[:, -1] * (columns.flatten() - x_c) / focal_length
        # t4 = timer.elapsed_seconds()
        points[:, 1] = points[:, -1] * (rows.flatten() - y_c) / focal_length
        # t5 = timer.elapsed_seconds()
        points = points[depth_image_flat > 0, :]
        # t6 = timer.elapsed_seconds()
        # Transform points into camera coordinate system (x-axis forward, z-axis up, y-axis left)
        quat1 = transformations.quaternion_about_axis(-np.pi / 2, [1, 0, 0])
        quat2 = transformations.quaternion_about_axis(-np.pi / 2, [0, 0, 1])
        quat = transformations.quaternion_multiply(quat2, quat1)
        # t7 = timer.elapsed_seconds()
        rot_mat = transformations.quaternion_matrix(quat)[:3, :3]
        # t8 = timer.elapsed_seconds()
        points_transformed = points.dot(rot_mat.T)
        # t9 = timer.elapsed_seconds()
        # print("Timing of _create_points_from_depth_image:")
        # print("  ", t1)
        # print("  ", t2 - t1)
        # print("  ", t3 - t2)
        # print("  ", t4 - t3)
        # print("  ", t5 - t4)
        # print("  ", t6 - t5)
        # print("  ", t7 - t6)
        # print("  ", t8 - t7)
        # print("  ", t9 - t8)
        # print("Total: ", t9)
        return points_transformed

    def _convert_normal_rgb_image_to_normal_image(self, normal_rgb_image, inplace=True):
        """Convert normal image in RGB encoding to normal image in vector encoding.
        RGB encoding means 128 color value is 0, 0 color value is -1, 255 color value is +1.
        """
        if inplace:
            assert(normal_rgb_image.dtype == np.float)
            normal_image = normal_rgb_image
        else:
            normal_image = np.array(normal_rgb_image, dtype=np.float)
        normal_image /= 0.5 * 255.
        normal_image -= 1.0
        # # For debugging. Show filtered depth image.
        # assert(not np.any(normal_image < -1.))
        # assert(not np.any(normal_image > 1.))
        # import cv2
        # normal_rgb_image_show = (normal_image + 1.0) * 0.5
        # cv2.imshow('normal', normal_rgb_image_show)
        # cv2.waitKey(10)
        return normal_image

    def get_view_direction_world(self):
        """Return camera viewing direction in world frame.
        Assuming x-axis points forward, y-axis left and z-axis up.
        """
        orientation_rpy = self.get_orientation_rpy()
        quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        x_axis = [1, 0, 0]
        view_direction = math_utils.rotate_vector_with_quaternion(quat, x_axis)
        return view_direction

    def get_ray_direction_image(self, width, height, focal_length):
        """Return camera ray for given width, height and focal length.
        Assuming x-axis points forward, y-axis left and z-axis up.
        """
        if self._ray_direction_image is not None:
            return self._ray_direction_image
        x_c = np.float(width) / 2 - 1
        y_c = np.float(height) / 2 - 1
        ray_directions = np.empty((height, width, 3))
        columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
        ray_directions[:, :, 0] = 1.
        # Sign swap because y is pointing to the left
        ray_directions[:, :, 1] = -(columns - x_c) / focal_length
        # Sign swap because of image y coordinate starting with 0 at the top
        ray_directions[:, :, 2] = -(rows - y_c) / focal_length
        # Normalize ray directions
        ray_directions /= np.sum(ray_directions ** 2, axis=2)[:, :, np.newaxis]
        # # For debugging. Show filtered depth image.
        # print("center ray:", ray_directions[height/2, width/2, :])
        # print("top ray:", ray_directions[0, width/2, :])
        # print("bottom ray:", ray_directions[height-1, width/2, :])
        # print("left ray:", ray_directions[height/2, 0, :])
        # print("right ray:", ray_directions[height/2, width-1, :])
        # import cv2
        # ray_directions_show = (-ray_directions + 1.0) * 0.5
        # cv2.imshow('ray_directions', ray_directions_show)
        # cv2.waitKey(10)
        return ray_directions

    def get_ray_direction_image_world(self, width, height, focal_length, orientation_rpy=None):
        """Return camera ray for given width, height and focal length in world frame.
        Assuming x-axis points forward, y-axis left and z-axis up.
        """
        ray_directions = self.get_ray_direction_image(width, height, focal_length)
        # Transform ray directions into world-frame
        if orientation_rpy is None:
            orientation_rpy = self.get_orientation_rpy()
        orientation_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        rot_mat = transformations.quaternion_matrix(orientation_quat)[:3, :3]
        ray_directions_world = ray_directions.dot(rot_mat.T)
        return ray_directions_world

    def get_depth_point_cloud_rpy(self, location, orientation_rpy, filter=True):
        """Return point cloud (from depth image) in local coordinate frame"""
        # timer = Timer()
        self.set_location(location)
        roll, pitch, yaw = orientation_rpy
        self.set_orientation_rpy(roll, pitch, yaw)
        focal_length = self.get_focal_length()
        # t1 = timer.elapsed_seconds()
        depth_image = self.get_depth_image()
        # Show depth image. Only for debugging.
        # import cv2
        # cv2.imshow("depth image", depth_image / 30.)
        # # cv2.imshow("depth image", depth_image / np.max(depth_image.flatten()))
        # cv2.waitKey(10)
        # t4 = timer.elapsed_seconds()
        rospy.logdebug("Depth image: min={}, max={}".format(
            np.min(depth_image.flatten()), np.max(depth_image.flatten())))
        if filter:
            normal_image = self.get_normal_image()
            # t5 = timer.elapsed_seconds()
            ray_directions = self.get_ray_direction_image_world(
                depth_image.shape[1], depth_image.shape[0],
                self.get_focal_length(), orientation_rpy)
            # t6 = timer.elapsed_seconds()
            self.filter_depth_image(depth_image, normal_image, ray_directions, inplace=True)
            # t7 = timer.elapsed_seconds()
            rospy.logdebug("Filtered depth image: min={}, max={}".format(
                np.min(depth_image.flatten()), np.max(depth_image.flatten())))
            # # For debugging. Show filtered depth image.
            # import cv2
            # depth_image_show = depth_image / 20.
            # depth_image_show[depth_image_show > 1] = 1
            # depth_image_show[depth_image_show < 0] = 0
            # cv2.imshow('depth', depth_image_show)
            # cv2.waitKey(10)
        # Create pointcloud message
        rospy.logdebug("Creating point cloud message")
        points = self._create_points_from_depth_image(depth_image, focal_length)
        # t8 = timer.elapsed_seconds()
        # print("Timing of get_depth_point_cloud_rpy():")
        # print("  ", t1)
        # print("  ", t2 - t1)
        # print("  ", t3 - t2)
        # print("  ", t4 - t3)
        # if filter:
        #     print("  ", t5 - t4)
        #     print("  ", t6 - t5)
        #     print("  ", t7 - t6)
        #     print("  ", t8 - t7)
        # else:
        #     print("  ", t8 - t4)
        # print("Total: ", t8)
        return points

    def get_depth_point_cloud_world_rpy(self, location, orientation_rpy, filter=True):
        """Return point cloud (from depth image) in world coordinate frame"""
        points = self.get_depth_point_cloud_rpy(location, orientation_rpy, filter=filter)
        world_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        world_trans = location
        world_rot_mat = transformations.quaternion_matrix(world_quat)[:3, :3]
        points_world = points.dot(world_rot_mat.T) + world_trans
        return points_world
