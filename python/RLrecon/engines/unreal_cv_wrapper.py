import os
import time
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

    def __init__(self,
                 address=None,
                 port=None,
                 image_scale_factor=0.5,
                 max_depth_distance=np.finfo(np.float).max,
                 max_depth_viewing_angle=math_utils.degrees_to_radians(90.),
                 max_request_trials=5,
                 request_timeout=5.0,
                 location_tolerance=1e-3,
                 orientation_tolerance=1e-3,
                 connect=True,
                 connect_wait_time=5.0,
                 connect_timeout=5.0):
        super(UnrealCVWrapper, self).__init__(max_depth_distance, max_depth_viewing_angle)
        self._width = None
        self._height = None
        self._image_scale_factor = image_scale_factor
        self._max_request_trials = max_request_trials
        self._request_timeout = request_timeout
        self._connect_wait_time = connect_wait_time
        self._connect_timeout = connect_timeout
        self._location_tolerance = location_tolerance
        self._orientation_tolerance = orientation_tolerance
        self._request_trials = 0
        if address is None:
            address = '127.0.0.1'
        if port is None:
            port = 9000
        self._cv_client = Client((address, port))
        if connect:
            self.connect()

    def connect(self):
        """Open connection to UnrealCV"""
        if self._cv_client.isconnected():
            print("WARNING: Already connected to UnrealCV")
        else:
            self._cv_client.connect(self._connect_timeout)
            if not self._cv_client.isconnected():
                raise(self.Exception("Unable to connect to UnrealCV"))

    def close(self):
        """Close connection to UnrealCV"""
        self._cv_client.disconnect()

    def _unrealcv_request(self, request):
        """Send a request to UnrealCV. Automatically retry in case of timeout."""
        result = None
        while result is None:
            self._request_trials += 1
            if self._request_trials > self._max_request_trials:
                raise self.Exception("UnrealCV request failed")
            result = self._cv_client.request(request, self._request_timeout)
            if result is None:
                print("UnrealCV request timed out. Retrying.")
                self._cv_client.disconnect()
                time.sleep(self._connect_wait_time)
                self._cv_client.connect()
        self._request_trials = 0
        return result

    def scale_image(self, image, scale_factor=None, interpolation_mode=cv2.INTER_CUBIC):
        """Scale an image to the desired size"""
        if scale_factor is None:
            scale_factor = self._image_scale_factor
        if scale_factor == 1:
            return image
        dsize = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        scaled_image = cv2.resize(image, dsize=dsize, interpolation=interpolation_mode)
        return scaled_image

    def scale_image_with_nearest_interpolation(self, image, scale_factor=None):
        """Scale an image to the desired size using 'nearest' interpolation"""
        return self.scale_image(image, scale_factor=scale_factor, interpolation_mode=cv2.INTER_NEAREST)

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

    def get_horizontal_field_of_view(self):
        """Return the horizontal field of view of the camera"""
        horz_fov_str = self._unrealcv_request('vget /camera/0/horizontal_fieldofview')
        horz_fov = float(horz_fov_str)
        # Convert to radians
        horz_fov = math_utils.degrees_to_radians(horz_fov)
        return horz_fov

    def set_horizontal_field_of_view(self, horz_fov):
        """Set the horizontal field of view of the camera in radians"""
        horz_fov = math_utils.radians_to_degrees(horz_fov)
        response = self._unrealcv_request('vset /camera/0/horizontal_fieldofview {:f}'.format(horz_fov))
        if response != "ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))

    def get_image_scale_factor(self):
        """Return scale factor for image retrieval"""
        return self._image_scale_factor

    def get_focal_length(self):
        """Return focal length of camera"""
        # # TODO: Focal length (and also projection matrix) should come from UnrealCV
        # return 320. * self._image_scale_factor
        try:
            horz_fov = self.get_horizontal_field_of_view()
        except UnrealCVWrapper.Exception:
            print("WARNING: UnrealCV does not support querying horizontal field of view. Assuming 90 degrees.")
            horz_fov = math_utils.degrees_to_radians(90.0)
        width = self.get_width()
        focal_length = width / (2 * np.tan(horz_fov / 2.))
        return focal_length

    def get_intrinsics(self):
        """Return intrinsics of camera"""
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.get_focal_length()
        intrinsics[1, 1] = self.get_focal_length()
        intrinsics[0, 2] = self.get_width() / 2.0
        intrinsics[1, 2] = self.get_height() / 2.0
        intrinsics[2, 2] = 1.0
        return intrinsics

    def get_rgb_image(self, scale_factor=None):
        """Return the current RGB image"""
        img_str = self._unrealcv_request('vget /camera/0/lit png')
        img = np.fromstring(img_str, np.uint8)
        rgb_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        rgb_image = self.scale_image(rgb_image, scale_factor)
        return rgb_image

    def get_rgb_image_by_file(self, scale_factor=None):
        """Return the current RGB image (transport via filesystem)"""
        filename = self._unrealcv_request('vget /camera/0/lit lit.png')
        rgb_image = cv2.imread(filename)
        rgb_image = self.scale_image(rgb_image, scale_factor)
        os.remove(filename)
        return rgb_image

    def get_normal_rgb_image(self, scale_factor=None):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)"""
        img_str = self._unrealcv_request('vget /camera/0/normal png')
        img = np.fromstring(img_str, np.uint8)
        normal_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        return normal_image

    def get_normal_rgb_image_by_file(self, scale_factor=None):
        """Return the current normal image in RGB encoding (i.e. 128 is 0, 0 is -1, 255 is +1)
        (transport via filesystem)
        """
        filename = self._unrealcv_request('vget /camera/0/normal normal.png')
        normal_image = cv2.imread(filename)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        os.remove(filename)
        return normal_image

    def get_normal_image(self, scale_factor=None):
        """Return the current normal image in vector representation"""
        normal_rgb_image = self.get_normal_rgb_image(scale_factor)
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        self.filter_normal_image(normal_image)
        return normal_image

    def get_normal_image_by_file(self, scale_factor=None):
        """Return the current normal image in vector representation (transport via filesystem)"""
        normal_rgb_image = self.get_normal_rgb_image_by_file(scale_factor)
        # Image is in BGR ordering, so [z, y, x]. Convert to RGB.
        normal_rgb_image = normal_rgb_image[:, :, ::-1]
        normal_image = self._convert_normal_rgb_image_to_normal_image(normal_rgb_image, inplace=False)
        # Convert left-handed Unreal system to right-handed system
        normal_image[:, :, 1] = -normal_image[:, :, 1]
        return normal_image

    def get_ray_distance_image(self, scale_factor=None):
        """Return the current ray-distance image"""
        img_str = self._unrealcv_request('vget /camera/0/depth npy')
        img_str_io = StringIO(img_str)
        ray_distance_image = np.load(img_str_io)
        ray_distance_image = self.scale_image_with_nearest_interpolation(ray_distance_image, scale_factor)
        return ray_distance_image

    def get_ray_distance_image_by_file(self, scale_factor=None):
        """Return the current ray-distance image (transport via filesystem)"""
        filename = self._unrealcv_request('vget /camera/0/depth depth.exr')
        ray_distance_image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        ray_distance_image = self.scale_image_with_nearest_interpolation(ray_distance_image, scale_factor)
        os.remove(filename)
        return ray_distance_image

    def get_depth_image(self, scale_factor=None):
        """Return the current depth image"""
        # timer = utils.Timer()
        ray_distance_image = self.get_ray_distance_image(scale_factor)
        depth_image = self._ray_distance_to_depth_image(ray_distance_image, self.get_focal_length())
        # print("get_depth_image() took {}".format(timer.elapsed_seconds())
        return depth_image

    def get_depth_image_by_file(self, scale_factor=None):
        """Return the current depth image (transport via filesystem)"""
        ray_distance_image = self.get_ray_distance_image(scale_factor)
        depth_image = self._ray_distance_to_depth_image_by_file(ray_distance_image, self.get_focal_length())
        return depth_image

    def get_location(self):
        """Return the current location in meters as [x, y, z]"""
        location_str = self._unrealcv_request('vget /camera/0/location')
        location_unreal = np.array([float(v) for v in location_str.split()])
        # Convert location from Unreal (cm) to meters
        location_unreal *= 0.01
        # Convert left-handed Unreal system to right-handed system
        location = math_utils.convert_xyz_from_left_to_right_handed(location_unreal)
        return location

    def get_orientation_rpy(self):
        """Return the current orientation in radians as [roll, pitch, yaw]"""
        orientation_str = self._unrealcv_request('vget /camera/0/rotation')
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

    def _is_location_set(self, location):
        current_location = self.get_location()
        if np.all(np.abs(current_location - location) < self._location_tolerance):
            return True
        return False

    def set_location(self, location, wait_until_set=False):
        """Set new location in meters as [x, y, z]"""
        # Convert right-handed system to left-handed Unreal system
        location_unreal = math_utils.convert_xyz_from_right_to_left_handed(location)
        # Convert meters to Unreal (cm)
        location_unreal *= 100
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_str = 'vset /camera/0/location {:f} {:f} {:f}'.format(
            location_unreal[0], location_unreal[1], location_unreal[2])
        # print("Sending location request: {}".format(request_str))
        response = self._unrealcv_request(request_str)
        if response != "ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_location_set(location):
                    return
            raise self.Exception("UnrealCV: New orientation was not set within time limit")

    def _is_orientation_rpy_set(self, roll, pitch, yaw):
        current_roll, current_pitch, current_yaw = self.get_orientation_rpy()
        if abs(current_roll - roll) < self._orientation_tolerance \
                and abs(current_pitch - pitch) < self._orientation_tolerance \
                and abs(current_yaw - yaw) < self._orientation_tolerance:
            return True
        return False

    def set_orientation_rpy(self, roll, pitch, yaw, wait_until_set=False):
        """Set new orientation in radians"""
        roll, pitch, yaw = math_utils.convert_rpy_from_right_to_left_handed([roll, pitch, yaw])
        # UnrealCV cannot handle scientific notation so we use :f format specifier
        request_str = 'vset /camera/0/rotation {:f} {:f} {:f}'.format(
            pitch * 180 / np.pi, yaw * 180 / np.pi, roll * 180 / np.pi)
        # print("Sending orientation request: {}".format(request_str))
        response = self._unrealcv_request(request_str)
        if response != "ok":
            raise self.Exception("UnrealCV request failed: {}".format(response))
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_orientation_rpy_set(roll, pitch, yaw):
                    return
            raise self.Exception("UnrealCV: New orientation was not set within time limit")

    def set_orientation_quat(self, quat):
        """Set new orientation quaterion quat = [w, x, y, z]"""
        # yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
        self.set_orientation_rpy(roll, pitch, yaw)

    def set_pose_quat(self, pose, wait_until_set=False):
        """Set new pose as a tuple of location and orientation quaternion"""
        self.set_location(pose[0])
        self.set_orientation_quat(pose[1])
        if wait_until_set:
            yaw, pitch, roll = transformations.euler_from_quaternion(pose[1], 'rzyx')
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_location_set(pose[0]) and self._is_orientation_rpy_set(yaw, pitch, roll):
                    return
            raise self.Exception("UnrealCV: New pose was not set within time limit")

    def set_pose_rpy(self, pose, wait_until_set=False):
        """Set new pose as a tuple of location and orientation rpy"""
        self.set_location(pose[0])
        self.set_orientation_rpy(*pose[1])
        if wait_until_set:
            start_time = time.time()
            while time.time() - start_time < self._request_timeout:
                if self._is_location_set(pose[0]) and self._is_orientation_rpy_set(*pose[1]):
                    return
            raise self.Exception("UnrealCV: New pose was not set within time limit")

    def test(self):
        """Perform some tests"""
        print("Performing some tests on UnrealCV")
        import time
        prev_depth_image = None
        location1 = self.get_location()
        location2 = location1 + [2, 2, 0]
        self.set_location(location1, wait_until_set=True)
        for i in xrange(100):
            self.set_location(location2, wait_until_set=True)
            _ = self.get_depth_image()
            self.set_location(location1, wait_until_set=True)
            depth_image = self.get_depth_image()
            if prev_depth_image is not None:
                assert(np.all(depth_image == prev_depth_image))
            prev_depth_image = depth_image
            time.sleep(0.1)