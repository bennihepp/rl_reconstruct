import numpy as np
from tf import transformations


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


"""Convert xyz position from left- to right-handed coordinate system"""
def convert_xyz_from_left_to_right_handed(location):
    location = np.array([location[0], -location[1], location[2]])
    return location


"""Convert xyz position from right- to left-handed coordinate system"""
def convert_xyz_from_right_to_left_handed(location):
    location = np.array([location[0], -location[1], location[2]])
    return location


"""Convert roll, pitch, yaw euler angles from left- to right-handed coordinate system"""
def convert_rpy_from_left_to_right_handed(orientation_rpy):
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


"""Convert roll, pitch, yaw euler angles from right- to left-handed coordinate system"""
def convert_rpy_from_right_to_left_handed(orientation_rpy):
    roll, pitch, yaw = orientation_rpy
    # Convert left-handed Unreal system to right-handed system
    yaw = -yaw
    pitch = -pitch
    return np.array([roll, pitch, yaw])


"""Convert roll, pitch, yaw euler angles to quaternion"""
def convert_rpy_to_quat(orientation_rpy):
    roll, pitch, yaw = orientation_rpy
    quat = transformations.quaternion_from_euler(yaw, pitch, roll, 'rzyx')
    return quat


"""Rotate a vector with a given quaternion"""
def rotate_vector_with_quaternion(quat, vec):
    vec_q = [vec[0], vec[1], vec[2], 0]
    rot_vec_q = transformations.quaternion_multiply(
        transformations.quaternion_multiply(quat, vec_q),
        transformations.quaternion_conjugate(quat))
    rot_vec = rot_vec_q[:3]
    return rot_vec
