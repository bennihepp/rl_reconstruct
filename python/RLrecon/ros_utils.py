import sys
import struct
import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import PointCloud2, PointField


def point_ros_to_numpy(point_msg, point_np=None):
    """Convert ROS Point message to numpy array"""
    if point_np is None:
        point_np = np.empty((3,))
    point_np[0] = point_msg.x
    point_np[1] = point_msg.y
    point_np[2] = point_msg.z
    return point_np


def point_numpy_to_ros(point_np, point_msg=None):
    """Convert numpy array to ROS Point message"""
    if point_msg is None:
        point_msg = Point()
    point_msg.x = point_np[0]
    point_msg.y = point_np[1]
    point_msg.z = point_np[2]
    return point_msg


def quaternion_ros_to_numpy(quat_msg, quat_np=None):
    """Convert ROS Quaternion message to numpy ([x, y, z, w])"""
    if quat_np is None:
        quat_np = np.empty((4,))
    quat_np[0] = quat_msg.x
    quat_np[1] = quat_msg.y
    quat_np[2] = quat_msg.z
    quat_np[3] = quat_msg.w
    return quat_np


def quaternion_numpy_to_ros(quat_np, quat_msg=None):
    """Convert numpy Quaternion ([x, y, z, w]) to ROS message"""
    if quat_msg is None:
        quat_msg = Quaternion()
    quat_msg.x = quat_np[0]
    quat_msg.y = quat_np[1]
    quat_msg.z = quat_np[2]
    quat_msg.w = quat_np[3]
    return quat_msg


_PC2_STRUCT_DATATYPES = {}
_PC2_STRUCT_DATATYPES[PointField.INT8]    = ('b', 1)
_PC2_STRUCT_DATATYPES[PointField.UINT8]   = ('B', 1)
_PC2_STRUCT_DATATYPES[PointField.INT16]   = ('h', 2)
_PC2_STRUCT_DATATYPES[PointField.UINT16]  = ('H', 2)
_PC2_STRUCT_DATATYPES[PointField.INT32]   = ('i', 4)
_PC2_STRUCT_DATATYPES[PointField.UINT32]  = ('I', 4)
_PC2_STRUCT_DATATYPES[PointField.FLOAT32] = ('f', 4)
_PC2_STRUCT_DATATYPES[PointField.FLOAT64] = ('d', 8)


_PC2_NUMPY_DATATYPES = {}
_PC2_NUMPY_DATATYPES[PointField.INT8]    = ('i', 1)
_PC2_NUMPY_DATATYPES[PointField.UINT8]   = ('u', 1)
_PC2_NUMPY_DATATYPES[PointField.INT16]   = ('i', 2)
_PC2_NUMPY_DATATYPES[PointField.UINT16]  = ('u', 2)
_PC2_NUMPY_DATATYPES[PointField.INT32]   = ('i', 4)
_PC2_NUMPY_DATATYPES[PointField.UINT32]  = ('u', 4)
_PC2_NUMPY_DATATYPES[PointField.FLOAT32] = ('f', 4)
_PC2_NUMPY_DATATYPES[PointField.FLOAT64] = ('f', 8)


def _point_cloud2_get_numpy_dtype(is_bigendian, fields):
    prefix = '>' if is_bigendian else '<'
    dtype = []
    offset = 0
    for field in sorted(fields, key=lambda f: f.offset):
        if field.datatype not in _PC2_NUMPY_DATATYPES:
            raise("Requesting non-supported PointField datatype {}".format(field.datatype))
        datatype_fmt, datatype_length = _PC2_NUMPY_DATATYPES[field.datatype]
        name = field.name
        for i in xrange(field.count):
            if field.count > 1:
                name = field.name + i
            dtype.append((name, prefix + "{}{}".format(datatype_fmt, datatype_length)))
        offset += field.count * datatype_length

    return dtype


def point_cloud2_ros_to_numpy(point_cloud_msg):
    """Convert ROS PointCloud2 message to structured numpy array"""
    np_dtype = _point_cloud2_get_numpy_dtype(point_cloud_msg.is_bigendian, point_cloud_msg.fields)
    itemsize = np.dtype(np_dtype).itemsize
    skip_bytes = point_cloud_msg.point_step - itemsize
    for i in xrange(skip_bytes):
        np_dtype.append(("_{}".format(i), "u1"))
    pc = np.fromstring(point_cloud_msg.data, np_dtype)
    return pc


def structured_to_3d_array(pc_arr):
    """Convert numpy structure array with x, y and z field to (n, 3) array"""
    pc_xyz = np.stack([pc_arr['x'], pc_arr['y'], pc_arr['z']], axis=1)
    return pc_xyz


def point_cloud2_numpy_to_ros(point_cloud, frame_id=None, timestamp=None):
    """Convert xyz numpy array to ROS PointCloud2 message"""
    assert(isinstance(point_cloud, np.ndarray))
    point_cloud = np.asarray(point_cloud, np.float32)
    point_cloud_msg = PointCloud2()
    if timestamp is None:
        timestamp = rospy.Time.now()
    point_cloud_msg.header.stamp = timestamp
    if frame_id is not None:
        point_cloud_msg.header.frame_id = frame_id
    point_cloud_msg.height = 1
    point_cloud_msg.width = point_cloud.shape[0]
    point_cloud_msg.fields = []
    offset = 0
    for i, name in enumerate(['x', 'y', 'z']):
        field = PointField()
        field.name = name
        field.offset = offset
        offset += np.float32().itemsize
        field.datatype = PointField.FLOAT32
        field.count = 1
        point_cloud_msg.fields.append(field)
    point_cloud_msg.is_bigendian = sys.byteorder == 'little'
    point_cloud_msg.point_step = offset
    point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
    point_cloud_msg.is_dense = True
    points_flat = point_cloud.flatten()
    point_cloud_msg.data = struct.pack("={}f".format(len(points_flat)), *points_flat)
    return point_cloud_msg
