import os
from cStringIO import StringIO
import numpy as np
import cv2
from unrealcv import Client
from engine import BaseEngine
from RLrecon import math_utils
from RLrecon.contrib import transformations
#from RLrecon import utils


class UnrealCVWrapper(BaseEngine):

    class Exception(RuntimeError):
        pass

    def __init__(self, address=None, port=None,
                 image_scale_factor=0.5,
                 max_depth_distance=np.finfo(np.float).max,
                 max_depth_viewing_angle=math_utils.degrees_to_radians(90.)):
        super(UnrealCVWrapper, self).__init__(max_depth_distance, max_depth_viewing_angle)
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
            raise(self.Exception("Unable to connect to UnrealCV"))

    def _scale_image(self, image, interpolation_mode=cv2.INTER_CUBIC):
        """Scale an image to the desired size"""
        dsize = (int(image.shape[1] * self._image_scale_factor), int(image.shape[0] * self._image_scale_factor))
        scaled_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_mode)
        return scaled_image

    def _scale_image_with_nearest_interpolation(self, image):
        """Scale an image to the desired size using 'nearest' interpolation"""
        return self._scale_image(image, interpolation_mode=cv2.INTER_NEAREST)

    def _ray_distance_to_depth_image(self, ray_distance_image, focal_length):
        """Convert a ray-distance image to a plane depth image"""
        height = ray_distance_image.shape[0]
        width = ray_distance_image.shape[1]
        x_c = np.float(width) / 2 - 1
        y_c = np.float(height) / 2 - 1
        columns, rows = np.meshgrid(np.linspace(0, width - 1, num=width), np.linspace(0, height - 1, num=height))
        dist_from_center = ((rows - y_c) ** 2 + (columns - x_c) ** 2) ** (0.5)
        depth_image = ray_distance_image / (1 + (dist_from_center / focal_length) ** 2) ** (0.5)
        return depth_image

    def close(self):
        """Close connection to UnrealCV"""
        self._cv_client.disconnect()

    def get_width(self):
        """Return width of image plane"""
        if self._width is None:
            rgb_image = self.get_rgb_image()
            self._height = rgb_image.shape[0]
            self._width = rgb_image.shape[1]
        return self._width

    def get_height(self):
        """Return height of image plane"""
        if self._height is None:
            self.get_width()
        return self._height

    def get_focal_length(self):
        """Return focal length of camera"""
        # TODO: Focal length (and also projection matrix) should come from UnrealCV
        return 320. * self._image_scale_factor

    def get_rgb_image(self):
        """Return the current RGB image"""
        img_str = self._cv_client.request('vget /camera/0/lit png')
        if img_str is None:
            raise self.Exception("UnrealCV request failed")
        img = np.fromstring(img_str, np.uint8)
        rgb_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        rgb_image = self._scale_image(rgb_image)
        return rgb_image

    def get_rgb_image_by_file(self):
        """Return the current RGB image (transport via filesystem)"""
        filename = self._cv_client.request('vget /camera/0/lit lit.png')
        if filename is None:
            raise self.Exception("UnrealCV request failed")
        rgb_image = cv2.imread(filename)
        rgb_image = self._scale_image(rgb_image)
        os.remove(filename)
        return rgb_image

    def get_normal_rgb_image(self):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)"""
        img_str = self._cv_client.request('vget /camera/0/normal png')
        if img_str is None:
            raise self.Exception("UnrealCV request failed")
        img = np.fromstring(img_str, np.uint8)
        normal_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        normal_image = self._scale_image_with_nearest_interpolation(normal_image)
        return normal_image

    def get_normal_rgb_image_by_file(self):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)
        (transport via filesystem)
        """
        filename = self._cv_client.request('vget /camera/0/normal normal.png')
        if filename is None:
            raise self.Exception("UnrealCV request failed")
        normal_image = cv2.imread(filename)
        normal_image = self._scale_image_with_nearest_interpolation(normal_image)
        os.remove(filename)
        return normal_image

    def get_normal_image(self):
        """Return the current normal image in vector representation"""
        normal_rgb_image = self.get_normal_rgb_image()
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        self.filter_normal_image(normal_image)
        return normal_image

    def get_normal_image_by_file(self):
        """Return the current normal image in vector representation (transport via filesystem)"""
        normal_rgb_image = self.get_normal_rgb_image_by_file()
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        return normal_image

    def get_ray_distance_image(self):
        """Return the current ray-distance image"""
        img_str = self._cv_client.request('vget /camera/0/depth npy')
        if img_str is None:
            raise self.Exception("UnrealCV request failed")
        img_str_io = StringIO(img_str)
        ray_distance_image = np.load(img_str_io)
        ray_distance_image = self._scale_image_with_nearest_interpolation(ray_distance_image)
        return ray_distance_image

    def get_ray_distance_image_by_file(self):
        """Return the current ray-distance image (transport via filesystem)"""
        filename = self._cv_client.request('vget /camera/0/depth depth.exr')
        if filename is None:
            raise self.Exception("UnrealCV request failed")
        ray_distance_image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        ray_distance_image = self._scale_image_with_nearest_interpolation(ray_distance_image)
        os.remove(filename)
        return ray_distance_image

    def get_depth_image(self):
        """Return the current depth image"""
        # timer = utils.Timer()
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        # print("get_depth_image() took {}".format(timer.elapsed_seconds())
        return depth_image

    def get_depth_image_by_file(self):
        """Return the current depth image (transport via filesystem)"""
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image_by_file(ray_distance_image, self.get_focal_length())
        return depth_image

    def get_location(self):
        """Return the current location in meters as [x, y, z]"""
        location_str = self._cv_client.request('vget /camera/0/location')
        if location_str is None:
            raise self.Exception("UnrealCV request failed")
        location_unreal = np.array([float(v) for v in location_str.split()])
        # Convert location from Unreal (cm) to meters
        location_unreal *= 0.01
        # Convert left-handed Unreal system to right-handed system
        location = math_utils.convert_xyz_from_left_to_right_handed(location_unreal)
        return location

    def get_orientation_rpy(self):
        """Return the current orientation in radians as [roll, pitch, yaw]"""
        orientation_str = self._cv_client.request('vget /camera/0/rotation')
        if orientation_str is None:
            raise self.Exception("UnrealCV request failed")
        pitch, yaw, roll = [float(v) * np.pi / 180. for v in orientation_str.split()]
        euler_rpy = np.array([roll, pitch, yaw])
        roll, pitch, yaw = math_utils.convert_rpy_from_left_to_right_handed(euler_rpy)
        if pitch <= -np.pi:
            pitch += 2 * np.pi
        elif pitch > np.pi:
            pitch -= 2 * np.pi
        euler_rpy = np.array([roll, pitch, yaw])
        return euler_rpy

    def get_orientation_quat(self):
        """Return the current orientation quaterion quat = [w, x, y, z]"""
        [roll, pitch, yaw] = self.get_orientation_rpy()
        quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')

        # # Transformation test for debugging
        # transform_mat = transformations.quaternion_matrix(quat)
        # sensor_x = transform_mat[:3, :3].dot(np.array([1, 0, 0]))
        # sensor_y = transform_mat[:3, :3].dot(np.array([0, 1, 0]))
        # sensor_z = transform_mat[:3, :3].dot(np.array([0, 0, 1]))
        # rospy.loginfo("sensor x axis: {} {} {}".format(sensor_x[0], sensor_x[1], sensor_x[2]))
        # rospy.loginfo("sensor y axis: {} {} {}".format(sensor_y[0], sensor_y[1], sensor_y[2]))
        # rospy.loginfo("sensor z axis: {} {} {}".format(sensor_z[0], sensor_z[1], sensor_z[2]))

        return quat

    def get_pose(self):
        """Return the current pose as a tuple of location and orientation quaternion"""
        return self.get_location(), self.get_orientation_quat()

    def set_location(self, location):
        """Set new location in meters as [x, y, z]"""
        # Convert right-handed system to left-handed Unreal system
        location_unreal = math_utils.convert_xyz_from_right_to_left_handed(location)
        # Convert meters to Unreal (cm)
        location_unreal *= 100
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_str = 'vset /camera/0/location {:f} {:f} {:f}'.format(
            location_unreal[0], location_unreal[1], location_unreal[2])
        # print("Sending location request: {}".format(request_str))
        response = self._cv_client.request(request_str)
        if response != "ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def set_orientation_rpy(self, roll, pitch, yaw):
        """Set new orientation in radians"""
        roll, pitch, yaw = math_utils.convert_rpy_from_right_to_left_handed([roll, pitch, yaw])
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_str = 'vset /camera/0/rotation {:f} {:f} {:f}'.format(
            pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi)
        # print("Sending orientation request: {}".format(request_str))
        response = self._cv_client.request(request_str)
        if response != "ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def set_orientation_quat(self, quat):
        """Set new orientation quaterion quat = [w, x, y, z]"""
        # yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
        self.set_orientation_rpy(roll, pitch, yaw)

    def set_pose(self, pose):
        """Set new pose as a tuple of location and orientation quaternion"""
        self.set_location(pose[0])
        self.set_rotation_quat(pose[1])
