import os
import time
from cStringIO import StringIO
import numpy as np
import cv2
from unrealcv import Client
from tf import transformations
from engine import BaseEngine
from RLrecon import math_utils
from RLrecon import utils


class UnrealCVWrapper(BaseEngine):

    class Exception(BaseException):
        pass

    def __init__(self, address=None, port=None, image_scale_factor=0.5):
        self._width = None
        self._height = None
        self._image_scale_factor = image_scale_factor
        if address is None:
            address = '127.0.0.1'
        if port is None:
            port = 9000
        self._cv_client = Client((address, port))
        self._cv_client.connect()
        if not self._cv_client.isconnected():
            raise(Exception("Unable to connect to UnrealCV"))

    """Scale an image to the desired size"""
    def _scale_image(self, image, interpolation_mode=cv2.INTER_CUBIC):
        dsize = (int(image.shape[1] * self._image_scale_factor), int(image.shape[0] * self._image_scale_factor))
        scaled_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_mode)
        return scaled_image

    """Scale an image to the desired size using 'nearest' interpolation"""
    def _scale_image_with_nearest_interpolation(self, image):
        return self._scale_image(image, interpolation_mode=cv2.INTER_NEAREST)

    """Convert a ray-distance image to a plane depth image"""
    def _ray_distance_to_depth_image(self, ray_distance_image, focal_length):
        height = ray_distance_image.shape[0]
        width = ray_distance_image.shape[1]
        x_c = np.float(width) / 2 - 1
        y_c = np.float(height) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
        dist_from_center = ((rows - y_c) ** 2 + (columns - x_c) ** 2) ** (0.5)
        depth_image = ray_distance_image / (1 + (dist_from_center / focal_length) ** 2) ** (0.5)
        return depth_image

    def close(self):
        self._cv_client.disconnect()

    """Return width of image plane"""
    def get_width(self):
        if self._width is None:
            rgb_image = self.get_rgb_image()
            self._height = rgb_image.shape[0]
            self._width = rgb_image.shape[1]
        return self._width

    """Return height of image plane"""
    def get_height(self):
        if self._height is None:
            self.get_width()
        return self._height

    """Return focal length of camera"""
    def get_focal_length(self):
        # TODO: Should come from UnrealCV
        return 320. * self._image_scale_factor

    """Return the current RGB image"""
    def get_rgb_image(self):
        img_str = self._cv_client.request('vget /camera/0/lit png')
        img = np.fromstring(img_str, np.uint8)
        rgb_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        rgb_image = self._scale_image(rgb_image)
        return rgb_image

    """Return the current RGB image (transport via filesystem)"""
    def get_rgb_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/lit lit.png')
        rgb_image = cv2.imread(filename)
        rgb_image = self._scale_image(rgb_image)
        os.remove(filename)
        return rgb_image

    """Return the current normal image"""
    def get_normal_image(self):
        img_str = self._cv_client.request('vget /camera/0/normal png')
        img = np.fromstring(img_str, np.uint8)
        normal_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        normal_image = self._scale_image_with_nearest_interpolation(normal_image)
        return normal_image

    """Return the current normal image (transport via filesystem)"""
    def get_normal_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/normal normal.png')
        normal_image = cv2.imread(filename)
        normal_image = self._scale_image_with_nearest_interpolation(normal_image)
        os.remove(filename)
        return normal_image

    """Return the current ray-distance image"""
    def get_ray_distance_image(self):
        img_str = self._cv_client.request('vget /camera/0/depth npy')
        img_str_io = StringIO(img_str)
        ray_distance_image = np.load(img_str_io)
        ray_distance_image = self._scale_image_with_nearest_interpolation(ray_distance_image)
        return ray_distance_image

    """Return the current ray-distance image (transport via filesystem)"""
    def get_ray_distance_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/depth depth.exr')
        ray_distance_image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        ray_distance_image = self._scale_image_with_nearest_interpolation(ray_distance_image)
        os.remove(filename)
        return ray_distance_image

    """Return the current depth image"""
    def get_depth_image(self):
        # timer = utils.Timer()
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        # print("get_depth_image() took {}".format(timer.elapsed_seconds())
        return depth_image

    """Return the current depth image (transport via filesystem)"""
    def get_depth_image_by_file(self):
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image_by_file(ray_distance_image, self.get_focal_length())
        return depth_image

    """Return the current location in meters as [x, y, z]"""
    def get_location(self):
        location_str = self._cv_client.request('vget /camera/0/location')
        location_unreal = np.array([float(v) for v in location_str.split()])
        # Convert location from Unreal (cm) to meters
        location_unreal *= 0.01
        # Convert left-handed Unreal system to right-handed system
        location = math_utils.convert_xyz_from_left_to_right_handed(location_unreal)
        return location

    """Return the current orientation in radians as [roll, pitch, yaw]"""
    def get_orientation_rpy(self):
        orientation_str = self._cv_client.request('vget /camera/0/rotation')
        pitch, yaw, roll = [float(v) * np.pi / 180. for v in orientation_str.split()]
        euler_rpy = np.array([roll, pitch, yaw])
        roll, pitch, yaw = math_utils.convert_rpy_from_left_to_right_handed(euler_rpy)
        if pitch <= -np.pi:
            pitch += 2 * np.pi
        elif pitch > np.pi:
            pitch -= 2 * np.pi
        euler_rpy = np.array([roll, pitch, yaw])
        return euler_rpy

    """Return the current orientation quaterion quat = [w, x, y, z]"""
    def get_orientation_quat(self):
        [roll, pitch, yaw] = self.get_orientation_rpy()
        # quat = transformations.quaternion_from_euler(roll, pitch, yaw, 'rxyz')
        quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')

        # # Transformation test
        # transform_mat = transformations.quaternion_matrix(quat)
        # sensor_x = transform_mat[:3, :3].dot(np.array([1, 0, 0]))
        # sensor_y = transform_mat[:3, :3].dot(np.array([0, 1, 0]))
        # sensor_z = transform_mat[:3, :3].dot(np.array([0, 0, 1]))
        # rospy.loginfo("sensor x axis: {} {} {}".format(sensor_x[0], sensor_x[1], sensor_x[2]))
        # rospy.loginfo("sensor y axis: {} {} {}".format(sensor_y[0], sensor_y[1], sensor_y[2]))
        # rospy.loginfo("sensor z axis: {} {} {}".format(sensor_z[0], sensor_z[1], sensor_z[2]))

        return quat

    """Return the current pose as a tuple of location and orientation quaternion"""
    def get_pose(self):
        return self.get_location(), self.get_orientation_quat()

    """Set new location in meters as [x, y, z]"""
    def set_location(self, location):
        # Convert right-handed system to left-handed Unreal system
        location_unreal = math_utils.convert_xyz_from_right_to_left_handed(location)
        # Convert meters to Unreal (cm)
        location_unreal *= 100
        request_str = 'vset /camera/0/location {} {} {}'.format(
            location_unreal[0], location_unreal[1], location_unreal[2])
        # print("Sending location request: {}".format(request_str))
        self._cv_client.request(request_str)

    """Set new orientation in radians"""
    def set_orientation_rpy(self, roll, pitch, yaw):
        roll, pitch, yaw = math_utils.convert_rpy_from_right_to_left_handed([roll, pitch, yaw])
        request_str = 'vset /camera/0/rotation {} {} {}'.format(
            pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi)
        # print("Sending orientation request: {}".format(request_str))
        self._cv_client.request(request_str)

    """Set new orientation quaterion quat = [w, x, y, z]"""
    def set_orientation_quat(self, quat):
        # yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
        self.set_orientation_rpy(roll, pitch, yaw)

    """Set new pose as a tuple of location and orientation quaternion"""
    def set_pose(self, pose):
        self.set_location(pose[0])
        self.set_rotation_quat(pose[1])
