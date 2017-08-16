import numpy as np
from RLrecon.contrib import transformations


def degrees_to_radians(degrees):
    """Convert angle in degrees to radians"""
    return degrees * np.pi / 180.0


def radians_to_degrees(radians):
    """Convert angle in radians to degrees"""
    return radians * 180.0 / np.pi


class BoundingBox(object):

    def __init__(self, min, max):
        self._min = np.array(min)
        self._max = np.array(max)

    def minimum(self):
        return self._min

    def maximum(self):
        return self._max

    def contains(self, xyz):
        return np.all(xyz >= self._min) and np.all(xyz <= self._max)

    def __str__(self):
        return "({}, {})".format(self.minimum(), self.maximum())


def convert_xyz_from_left_to_right_handed(location):
    """Convert xyz position from left- to right-handed coordinate system"""
    location = np.array([location[0], -location[1], location[2]])
    return location


def convert_xyz_from_right_to_left_handed(location):
    """Convert xyz position from right- to left-handed coordinate system"""
    location = np.array([location[0], -location[1], location[2]])
    return location


def convert_rpy_from_left_to_right_handed(orientation_rpy):
    """Convert roll, pitch, yaw euler angles from left- to right-handed coordinate system"""
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


def convert_rpy_from_right_to_left_handed(orientation_rpy):
    """Convert roll, pitch, yaw euler angles from right- to left-handed coordinate system"""
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


def convert_rpy_to_quat(orientation_rpy):
    """Convert roll, pitch, yaw euler angles to quaternion"""
    roll, pitch, yaw = orientation_rpy
    quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
    return quat


def rotate_vector_with_quaternion(quat, vec):
    """Rotate a vector with a given quaternion (x, y, z, w)"""
    vec_q = [vec[0], vec[1], vec[2], 0]
    rot_vec_q = transformations.quaternion_multiply(
        transformations.quaternion_multiply(quat, vec_q),
        transformations.quaternion_conjugate(quat))
    rot_vec = rot_vec_q[:3]
    return rot_vec


def rotate_vector_with_rpy(orientation_rpy, vec):
    """Rotate a vector with a given rpy orientation (roll, pitch, yaw)"""
    quat = convert_rpy_to_quat(orientation_rpy)
    return rotate_vector_with_quaternion(quat, vec)


def is_vector_equal(vec1, vec2, tolerance=1e-10):
    """Compare if two vectors are equal (L1-norm) according to a tolerance"""
    return np.all(np.abs(vec1 - vec2) <= tolerance)


def is_quaternion_equal(quat1, quat2, tolerance=1e-10):
    """Compare if two quaternions are equal

    This depends on L1-norm. A better way would be to use the angular difference
    """
    return is_vector_equal(quat1, quat2, tolerance) \
        or is_vector_equal(quat1, -quat2, tolerance)


class SinglePassMeanAndVariance(object):

    def __init__(self, size=None):
        self._mean = np.zeros(size)
        self._variance_acc = np.zeros(size)
        self._N = 0

    def add_value(self, value):
        self._N += 1
        prev_mean = self._mean
        self._mean += (value - self._mean) / self._N
        self._variance_acc += (value - self._mean) * (value - prev_mean)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance_acc / float(self._N - 1)

    @property
    def stddev(self):
        return np.sqrt(self.variance)

    @property
    def num_samples(self):
        return self._N
