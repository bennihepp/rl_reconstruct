import os
from cStringIO import StringIO
import numpy as np
import cv2
from unrealcv import Client
from tf import transformations


class UnrealCVWrapper:

    class Exception(BaseException):
        pass

    class RaycastResult(object):

        def __init__(self):
            self.expected_reward = 0

    def __init__(self, address=None, port=None):
        if address is None:
            address = '127.0.0.1'
        if port is None:
            port = 9000
        self._cv_client = Client((address, port))
        self._cv_client.connect()
        if not self._cv_client.isconnected():
            raise(Exception("Unable to connect to UnrealCV"))

    def close(self):
        self._cv_client.disconnect()

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

    """Returns focal length of camera"""
    def get_focal_length(self):
        return 320.

    """Returns the current RGB image"""
    def get_rgb_image(self):
        img_str = self._cv_client.request('vget /camera/0/lit png')
        img = np.fromstring(img_str, np.uint8)
        rgb_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return rgb_image

    """Returns the current RGB image (transport via filesystem)"""
    def get_rgb_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/lit lit.png')
        rgb_image = cv2.imread(filename)
        os.remove(filename)
        return rgb_image

    """Returns the current normal image"""
    def get_normal_image(self):
        img_str = self._cv_client.request('vget /camera/0/normal png')
        img = np.fromstring(img_str, np.uint8)
        normal_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return normal_image

    """Returns the current normal image (transport via filesystem)"""
    def get_normal_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/normal normal.png')
        normal_image = cv2.imread(filename)
        os.remove(filename)
        return normal_image

    """Returns the current ray-distance image"""
    def get_ray_distance_image(self):
        img_str = self._cv_client.request('vget /camera/0/depth npy')
        img_str_io = StringIO(img_str)
        ray_distance_image = np.load(img_str_io)
        return ray_distance_image

    """Returns the current ray-distance image (transport via filesystem)"""
    def get_ray_distance_image_by_file(self):
        filename = self._cv_client.request('vget /camera/0/depth depth.exr')
        ray_distance_image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        os.remove(filename)
        return ray_distance_image

    """Returns the current depth image"""
    def get_depth_image(self):
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        return depth_image

    """Returns the current depth image (transport via filesystem)"""
    def get_depth_image_by_file(self):
        ray_distance_image = self.get_ray_distance_image()
        depth_image = self._ray_distance_to_depth_image_by_file(ray_distance_image, self.get_focal_length())
        return depth_image

    """Returns the current location in meters as [x, y, z]"""
    def get_location(self):
        location_str = self._cv_client.request('vget /camera/0/location')
        location = np.array([float(v) for v in location_str.split()])
        # Convert location from Unreal (cm) to meters
        location *= 0.01
        # Convert left-handed Unreal system to right-handed system
        location_z_up = np.array([location[0], -location[1], location[2]])
        return location_z_up

    """Returns the current orientation in radians as [roll, pitch, yaw]"""
    def get_orientation_rpy(self):
        orientation_str = self._cv_client.request('vget /camera/0/rotation')
        pitch, yaw, roll = [float(v) * np.pi / 180. for v in orientation_str.split()]
        euler_rpy = np.array([roll, pitch, yaw])
        return euler_rpy

    """Returns the current orientation quaterion quat = [w, x, y, z]"""
    def get_orientation_quat(self):
        [roll, pitch, yaw] = self.get_orientation_rpy()
        # Convert left-handed Unreal system to right-handed system
        yaw = -yaw
        pitch = -pitch
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
        location_unreal = np.array([location[0], -location[1], location[2]])
        # Convert meters to Unreal (cm)
        location_unreal *= 100
        request_str = 'vset /camera/0/location {} {} {}'.format(
            location_unreal[0], location_unreal[1], location_unreal[2])
        # print("Sending location request: {}".format(request_str))
        self._cv_client.request(request_str)

    """Set new orientation in radians"""
    def set_orientation_rpy(self, roll, pitch, yaw):
        request_str = 'vset /camera/0/rotation {} {} {}'.format(
            pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi)
        # print("Sending orientation request: {}".format(request_str))
        self._cv_client.request(request_str)

    """Set new orientation quaterion quat = [w, x, y, z]"""
    def set_orientation_quat(self, quat):
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        self.set_orientation_rpy(roll, pitch, yaw)

    """Set new pose as a tuple of location and orientation quaternion"""
    def set_pose(self, pose):
        self.set_location(pose[0])
        self.set_rotation_quat(pose[1])
