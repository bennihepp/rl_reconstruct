import time
import numpy as np
import cv2
from pybh import math_utils
from pybh import zmq_utils
from pybh import serialization
from pybh import log_utils
from pybh.contrib import transformations
from .engine import BaseEngine


logger = log_utils.get_logger("RLrecon/mesh_renderer_zmq_client")


class MeshRendererZMQClient(BaseEngine):

    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_CUBIC = cv2.INTER_CUBIC

    class Exception(RuntimeError):
        pass

    def __init__(self,
                 address="tcp://localhost:22222",
                 image_scale_factor=1.0,
                 max_depth_distance=np.finfo(np.float).max,
                 max_depth_viewing_angle=math_utils.degrees_to_radians(90.),
                 max_request_trials=3,
                 request_timeout=0.5,
                 location_tolerance=1e-3,
                 orientation_tolerance=1e-3,
                 connect=True,
                 connect_wait_time=0.5,
                 connect_timeout=0.5,
                 keep_pose_state=False):
        super(MeshRendererZMQClient, self).__init__(max_depth_distance, max_depth_viewing_angle)
        self._serializer = serialization.MsgPackSerializer()
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
        self._conn = zmq_utils.Connection(address, zmq_utils.zmq.REQ)
        if keep_pose_state:
            self._location = np.array([0, 0, 0], dtype=np.float)
            self._orientation_rpy = np.array([0, 0, 0], dtype=np.float)
        self._keep_pose_state = keep_pose_state
        if connect:
            self.connect()

    def _is_location_set(self, location):
        current_location = self.get_location()
        if np.all(np.abs(current_location - location) < self._location_tolerance):
            return True
        return False

    def _is_orientation_rpy_set(self, roll, pitch, yaw):
        current_roll, current_pitch, current_yaw = self.get_orientation_rpy()
        if math_utils.is_angle_equal(current_roll, roll, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_pitch, pitch, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_yaw, yaw, self._orientation_tolerance):
            return True
        return False

    def _is_pose_rpy_set(self, pose):
        location = pose[0]
        roll, pitch, yaw = pose[1]
        current_location, current_euler_rpy = self.get_pose_rpy()
        current_roll, current_pitch, current_yaw = current_euler_rpy
        # logger.info("Desired pose:", location, roll, pitch, yaw)
        # logger.info("Current pose:", current_location, current_roll, current_pitch, current_yaw)
        if np.all(np.abs(current_location - location) < self._location_tolerance) \
                and math_utils.is_angle_equal(current_roll, roll, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_pitch, pitch, self._orientation_tolerance) \
                and math_utils.is_angle_equal(current_yaw, yaw, self._orientation_tolerance):
            return True
        return False

    def _send(self, msg):
        msg_dump = self._serializer.dumps(msg)
        self._conn.send(msg_dump)

    def _recv(self, timeout=None):
        if timeout is None:
            timeout = self._request_timeout
        # ZMQ takes timeout in milli-seconds
        msg_dump = self._conn.recv(timeout=timeout * 1000)
        if msg_dump is None:
            return None
        msg = self._serializer.loads(msg_dump)
        return msg

    def _send_recv(self, msg, timeout=None):
        self._send(msg)
        return self._recv(timeout)

    def _send_recv_retry(self, msg, timeout=None):
        """Send a request to renderer. Automatically retry in case of timeout."""
        response = None
        while response is None:
            self._request_trials += 1
            if self._request_trials > self._max_request_trials:
                raise self.Exception("Request failed")
            response = self._send_recv(msg, timeout)
            if response is None:
                logger.warn("Renderer request timed out. Retrying.")
                self._conn.reconnect()
                time.sleep(self._connect_wait_time)
        self._request_trials = 0
        return response

    def _request(self, request_name, msg_dict, timeout=None):
        request_dict = {b"_request": request_name}
        request_dict.update(msg_dict)
        response = self._send_recv_retry(request_dict, timeout)
        if b"_error" in response:
            raise self.Exception("ERROR on request: {}".format(response[b"error"]))
        return response

    def _request_get(self, names, timeout=None):
        if not isinstance(names, list):
            names = [names]
        response = self._request(b"get", {b"names": names}, timeout)
        values = {name: response[name] for name in names}
        return values

    def _request_render_images(self, requested_images, use_trackball=False, timeout=None):
        if not isinstance(requested_images, list):
            requested_images = [requested_images]
        request_dict = {b"requested_images": requested_images}
        if use_trackball:
            request_dict[b"use_trackball"] = True
        if self._keep_pose_state:
            request_dict[b"location"] = self._location
            request_dict[b"orientation_rpy"] = self._orientation_rpy
        response = self._request(b"render_images", request_dict, timeout)
        images = {image_name: response[image_name] for image_name in requested_images}
        if b"depth_image" in images:
            depth_image = images[b"depth_image"]
            # Make sure no ray is longer than max depth distance
            depth_image = np.minimum(depth_image, self._max_depth_distance)
            # Set invalid rays (i.e. no scene content to max depth distance)
            depth_image[depth_image <= 0] = self._max_depth_distance
            images[b"depth_image"] = depth_image
        return images

    def _request_set(self, value_dict, timeout=None):
        self._request(b"set", value_dict, timeout)

    def connect(self):
        """Open connection to renderer"""
        if self._conn.is_connected():
            logger.warn("WARNING: Already connected to renderer")
        else:
            self._conn.connect()
            logger.debug("Requesting ping ...")
            response = self._request(b"ping", {}, self._connect_timeout)
            logger.debug("Received pong: {}".format(response))
            if response is None:
                raise(self.Exception("Unable to connect to renderer"))
            assert response[b"ping"] == b"pong"

    def close(self):
        """Close connection to renderer"""
        self._conn.disconnect()

    def get_width(self):
        """Return width of image plane"""
        if self._width is None:
            values = self._request_get([b"width", b"height"])
            self._width = values[b"width"]
            self._height = values[b"height"]
        return self._width

    def get_height(self):
        """Return height of image plane"""
        if self._height is None:
            self.get_width()
        return self._height

    def get_horizontal_field_of_view_degrees(self):
        """Return the horizontal field of view of the camera"""
        values = self._request_get([b"horz_fov"])
        horz_fov = values[b"horz_fov"]
        return horz_fov

    def get_horizontal_field_of_view(self):
        """Return the horizontal field of view of the camera"""
        horz_fov_degrees = self.get_horizontal_field_of_view_degrees()
        # Convert to radians
        horz_fov = math_utils.degrees_to_radians(horz_fov_degrees)
        return horz_fov

    def set_horizontal_field_of_view(self, horz_fov):
        """Set the horizontal field of view of the camera in radians"""
        horz_fov = math_utils.radians_to_degrees(horz_fov)
        self._request_set({b"horz_fov": horz_fov})

    def get_focal_length(self):
        """Return focal length of camera"""
        horz_fov = self.get_horizontal_field_of_view()
        width = self.get_width()
        focal_length = width / (2 * np.tan(horz_fov / 2.))
        return focal_length

    def get_intrinsics(self):
        """Return intrinsics of camera"""
        intrinsics = np.zeros((3, 3))
        intrinsics[0, 0] = self.get_focal_length()
        intrinsics[1, 1] = self.get_focal_length()
        intrinsics[0, 2] = (self.get_width() - 1) / 2.0
        intrinsics[1, 2] = (self.get_height() - 1) / 2.0
        intrinsics[2, 2] = 1.0
        return intrinsics

    def get_rgb_image(self, scale_factor=None, use_trackball=False):
        """Return the current RGB image"""
        response = self._request_render_images([b"rgb_image"], use_trackball=use_trackball)
        rgb_image = response[b"rgb_image"]
        rgb_image = self.scale_image(rgb_image, scale_factor)
        rgb_image = self.convert_rgba_to_bgra(rgb_image)
        return rgb_image

    def get_normal_image(self, scale_factor=None, use_trackball=False):
        """Return the current normal image in vector representation"""
        response = self._request_render_images([b"normal_image"], use_trackball=use_trackball)
        normal_image = response[b"normal_image"]
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        self.filter_normal_image(normal_image)
        normal_image = self.convert_rgb_to_bgr(normal_image)
        return normal_image

    def get_depth_image(self, scale_factor=None, use_trackball=False):
        """Return the current depth image"""
        response = self._request_render_images([b"depth_image"], use_trackball=use_trackball)
        depth_image = response[b"depth_image"]
        depth_image = self.scale_image_with_nearest_interpolation(depth_image, scale_factor)
        return depth_image

    def get_rgb_depth_normal_images(self, scale_factor=None, use_trackball=False):
        """Return the current color, normal and depth image"""
        response = self._request_render_images([b"rgb_image", b"normal_image", b"depth_image"], use_trackball=use_trackball)
        rgb_image = response[b"rgb_image"]
        normal_image = response[b"normal_image"]
        depth_image = response[b"depth_image"]
        rgb_image = self.scale_image(rgb_image, scale_factor)
        rgb_image = self.convert_rgba_to_bgra(rgb_image)
        depth_image = self.scale_image_with_nearest_interpolation(depth_image, scale_factor)
        normal_image = self.scale_image_with_nearest_interpolation(normal_image, scale_factor)
        normal_image = self.convert_rgb_to_bgr(normal_image)
        return rgb_image, depth_image, normal_image

    def get_location(self):
        """Return the current location in meters as [x, y, z]"""
        if self._keep_pose_state:
            return self._location.copy()
        response = self._request_get([b"location"])
        location = response[b"location"]
        return location

    def get_orientation_rpy(self):
        """Return the current orientation in radians as [roll, pitch, yaw]"""
        if self._keep_pose_state:
            orientation_rpy = self._orientation_rpy
        else:
            response = self._request_get([b"orientation_rpy"])
            orientation_rpy = response[b"orientation_rpy"]
        # Convert to radians
        orientation_rpy = [math_utils.degrees_to_radians(v) for v in orientation_rpy]
        roll, pitch, yaw = orientation_rpy
        if pitch <= -np.pi:
            pitch += 2 * np.pi
        elif pitch > np.pi:
            pitch -= 2 * np.pi
        orientation_rpy = np.array([roll, pitch, yaw])
        return orientation_rpy

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

    def get_pose_rpy(self):
        """Return the current pose as a tuple of location and orientation as [roll, pitch, yaw]"""
        if self._keep_pose_state:
            orientation_rpy = [math_utils.degrees_to_radians(v) for v in self._orientation_rpy]
            return self._location.copy(), orientation_rpy
        response = self._request_get([b"location", b"orientation_rpy"])
        location = response[b"location"]
        orientation_rpy = response[b"orientation_rpy"]
        # Convert to radians
        orientation_rpy = [math_utils.degrees_to_radians(v) for v in orientation_rpy]
        return location, orientation_rpy

    def get_pose_quat(self):
        """Return the current pose as a tuple of location and orientation quaternion"""
        location, orientation_rpy = self.get_pose_rpy()
        roll, pitch, yaw = orientation_rpy
        if pitch <= -np.pi:
            pitch += 2 * np.pi
        elif pitch > np.pi:
            pitch -= 2 * np.pi
        quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
        return location, quat

    def _is_location_set(self, location):
        current_location = self.get_location()
        if np.all(np.abs(current_location - location) < self._location_tolerance):
            return True
        return False

    def set_location(self, location, wait_until_set=False):
        """Set new location in meters as [x, y, z]"""
        if self._keep_pose_state:
            self._location[:] = location
        else:
            self._request_set({b"location": location})
            if wait_until_set:
                if not self._is_location_set(location):
                    raise self.Exception("New orientation was not set")

    def _is_orientation_rpy_set(self, roll, pitch, yaw):
        current_roll, current_pitch, current_yaw = self.get_orientation_rpy()
        if abs(current_roll - roll) < self._orientation_tolerance \
                and abs(current_pitch - pitch) < self._orientation_tolerance \
                and abs(current_yaw - yaw) < self._orientation_tolerance:
            return True
        return False

    def set_orientation_rpy(self, roll, pitch, yaw, wait_until_set=False):
        """Set new orientation in radians"""
        if self._keep_pose_state:
            orientation_rpy = [roll, pitch, yaw]
            orientation_rpy = [math_utils.radians_to_degrees(v) for v in orientation_rpy]
            self._orientation_rpy[:] = orientation_rpy
        else:
            orientation_rpy = [roll, pitch, yaw]
            # Convert to degrees
            orientation_rpy = [math_utils.radians_to_degrees(v) for v in orientation_rpy]
            self._request_set({b"orientation_rpy": orientation_rpy})
            if wait_until_set:
                if not self._is_orientation_rpy_set(roll, pitch, yaw):
                    raise self.Exception("New orientation was not set")

    def set_orientation_quat(self, quat):
        """Set new orientation quaterion quat = [w, x, y, z]"""
        # yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rxyz')
        yaw, pitch, roll = transformations.euler_from_quaternion(quat, 'rzyx')
        self.set_orientation_rpy(roll, pitch, yaw)

    def set_pose_rpy(self, pose, wait_until_set=False):
        if self._keep_pose_state:
            self._location[:] = pose[0]
            orientation_rpy = pose[1]
            orientation_rpy = [math_utils.radians_to_degrees(v) for v in orientation_rpy]
            self._orientation_rpy[:] = orientation_rpy
        else:
            location = pose[0]
            orientation_rpy = pose[1]
            # Convert to degrees
            orientation_rpy = [math_utils.radians_to_degrees(v) for v in orientation_rpy]
            """Set new pose as a tuple of location and orientation rpy"""
            self._request_set({b"location": location, b"orientation_rpy": orientation_rpy})
            if wait_until_set:
                if not self._is_pose_rpy_set(pose):
                    raise self.Exception("New pose was not set: new pose={}, current pose={}".format(
                        pose, self.get_pose_rpy()
                    ))

    def set_pose_quat(self, pose, wait_until_set=False):
        """Set new pose as a tuple of location and orientation quaternion"""
        location = pose[0]
        yaw, pitch, roll = transformations.euler_from_quaternion(pose[1], 'rzyx')
        orientation_rpy = [roll, pitch, yaw]
        self.set_pose_rpy((location, orientation_rpy, wait_until_set))

    def enable_input(self):
        self._request_set({b"input_enabled": True})

    def disable_input(self):
        self._request_set({b"input_enabled": False})

    def set_window_active(self, window_active):
        self._request_set({b"window_active": window_active})

    def set_window_visible(self, window_visible):
        self._request_set({b"window_visible": window_visible})

    def test(self):
        """Perform some tests"""
        logger.info("Performing some tests on renderer")
        import time
        prev_depth_image = None
        location1 = self.get_location()
        location2 = location1 + [2, 2, 0]
        self.set_location(location1, wait_until_set=True)
        for i in range(100):
            self.set_location(location2, wait_until_set=True)
            _ = self.get_depth_image()
            self.set_location(location1, wait_until_set=True)
            depth_image = self.get_depth_image()
            if prev_depth_image is not None:
                assert(np.all(depth_image == prev_depth_image))
            prev_depth_image = depth_image
            time.sleep(0.1)
