from __future__ import print_function
import sys
import struct
import numpy as np
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import Transform
from sensor_msgs.msg import PointCloud2, PointField
from std_srvs.srv import Empty, EmptyRequest
from octomap_server_ext.srv import SetScoreBoundingBox, SetScoreBoundingBoxRequest
from octomap_server_ext.srv import InsertPointCloud, InsertPointCloudRequest
from octomap_server_ext.srv import InsertDepthMap, InsertDepthMapRequest
from octomap_server_ext.msg import Ray
from octomap_server_ext.srv import Info, InfoRequest
from octomap_server_ext.srv import QueryVoxels, QueryVoxelsRequest
from octomap_server_ext.srv import QuerySubvolume, QuerySubvolumeRequest
from octomap_server_ext.srv import Raycast, RaycastRequest
from octomap_server_ext.srv import RaycastCamera, RaycastCameraRequest
from octomap_server_ext.srv import ClearBoundingBox, ClearBoundingBoxRequest
from octomap_server_ext.srv import OverrideBoundingBox, OverrideBoundingBoxRequest
from RLrecon import math_utils, ros_utils
from RLrecon.utils import Timer


class OctomapExtMapper(object):

    class Ray(object):

        def __init__(self, origin, direction):
            self._origin = origin
            self._direction = direction

        def origin(self):
            return self._origin

        def direction(self):
            return self._direction

    class RaycastResultPoint(object):

        def __init__(self, xyz, occupancy, observation_certainty, is_surface, is_known):
            self.xyz = xyz
            self.occupancy = occupancy
            self.observation_certainty = observation_certainty
            self.is_surface = is_surface
            self.is_known = is_known

    class Exception(RuntimeError):

        def __init__(self, msg):
            super(RuntimeError, self).__init__(msg)

    def __init__(self,
                 set_score_bounding_box_topic='/octomap_server_ext/set_score_bounding_box',
                 insert_point_cloud_topic='/octomap_server_ext/insert_point_cloud',
                 insert_depth_map_topic='/octomap_server_ext/insert_depth_map',
                 info_topic='/octomap_server_ext/info',
                 query_voxels_topic='/octomap_server_ext/query_voxels',
                 query_subvolume_topic='/octomap_server_ext/query_subvolume',
                 raycast_topic='/octomap_server_ext/raycast',
                 raycast_camera_topic='/octomap_server_ext/raycast_camera',
                 raycast_point_cloud_pub_topic='/octomap_server_ext/raycast_point_cloud',
                 reset_topic='/octomap_server_ext/reset',
                 clear_bounding_box_topic='/octomap_server_ext/clear_bounding_box',
                 override_bounding_box_topic='/octomap_server_ext/override_bounding_box',
                 world_frame_id='map'):
        self._set_score_bounding_box_topic = set_score_bounding_box_topic
        self._insert_point_cloud_topic = insert_point_cloud_topic
        self._insert_depth_map_topic = insert_depth_map_topic
        self._info_topic = info_topic
        self._query_voxels_topic = query_voxels_topic
        self._query_subvolume_topic = query_subvolume_topic
        self._raycast_topic = raycast_topic
        self._raycast_camera_topic = raycast_camera_topic
        self._reset_topic = reset_topic
        self._clear_bounding_box_topic = clear_bounding_box_topic
        self._override_bounding_box_topic = override_bounding_box_topic
        if raycast_point_cloud_pub_topic is not None and len(raycast_point_cloud_pub_topic) > 0:
          self._raycast_point_cloud_pub = rospy.Publisher(raycast_point_cloud_pub_topic, PointCloud2, queue_size=1)
        else:
          self._raycast_point_cloud_pub = None
        self._world_frame_id = world_frame_id
        self._connect_services()

    def _connect_services(self):
        self._connect_set_score_bounding_box_service()
        self._connect_insert_point_cloud_service()
        self._connect_insert_depth_map_service()
        self._connect_info_service()
        self._connect_query_voxels_service()
        self._connect_query_subvolume_service()
        self._connect_raycast_service()
        self._connect_raycast_camera_service()
        self._connect_reset()
        self._connect_clear_bounding_box()
        self._connect_override_bounding_box()

    def _connect_set_score_bounding_box_service(self):
        self._set_score_bounding_box_service = rospy.ServiceProxy(self._set_score_bounding_box_topic, SetScoreBoundingBox, persistent=True)

    def _connect_insert_point_cloud_service(self):
        self._insert_point_cloud_service = rospy.ServiceProxy(self._insert_point_cloud_topic, InsertPointCloud, persistent=True)

    def _connect_insert_depth_map_service(self):
        self._insert_depth_map_service = rospy.ServiceProxy(self._insert_depth_map_topic, InsertDepthMap, persistent=True)

    def _connect_info_service(self):
        self._info_service = rospy.ServiceProxy(self._info_topic, Info, persistent=True)

    def _connect_query_voxels_service(self):
        self._query_voxels_service = rospy.ServiceProxy(self._query_voxels_topic, QueryVoxels, persistent=True)

    def _connect_query_subvolume_service(self):
        self._query_subvolume_service = rospy.ServiceProxy(self._query_subvolume_topic, QuerySubvolume, persistent=True)

    def _connect_raycast_service(self):
        self._raycast_service = rospy.ServiceProxy(self._raycast_topic, Raycast, persistent=True)

    def _connect_raycast_camera_service(self):
        self._raycast_camera_service = rospy.ServiceProxy(self._raycast_camera_topic, RaycastCamera, persistent=True)

    def _connect_reset(self):
        self._reset_service = rospy.ServiceProxy(self._reset_topic, Empty, persistent=True)

    def _connect_clear_bounding_box(self):
        self._clear_bounding_box_service = rospy.ServiceProxy(self._clear_bounding_box_topic, ClearBoundingBox, persistent=True)

    def _connect_override_bounding_box(self):
        self._override_bounding_box_service = rospy.ServiceProxy(self._override_bounding_box_topic, OverrideBoundingBox, persistent=True)

    def _create_point_cloud_msg(self, point_cloud, frame_id='depth_sensor', timestamp=rospy.Time.now()):
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = timestamp
        point_cloud_msg.header.frame_id = frame_id
        point_cloud_msg.height = 1
        # point_cloud_msg.width = len(points)
        point_cloud_msg.width = point_cloud.shape[0]
        point_cloud_msg.fields = []
        for i, name in enumerate(['x', 'y', 'z']):
            field = PointField()
            field.name = name
            field.offset = i * struct.calcsize('f')
            field.datatype = PointField.FLOAT32
            field.count = 1
            point_cloud_msg.fields.append(field)
        point_cloud_msg.is_bigendian = sys.byteorder == 'little'
        point_cloud_msg.point_step = 3 * struct.calcsize('f')
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.is_dense = True
        points_flat = point_cloud.flatten()
        point_cloud_msg.data = struct.pack("={}f".format(len(points_flat)), *points_flat)
        return point_cloud_msg

    def _get_transform_msg_rpy(self, location, orientation_rpy):
        quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        return self._get_transform_msg_quat(location, quat)

    def _get_transform_msg_quat(self, location, orientation_quat):
        # Create transform message
        transform = Transform()
        ros_utils.point_numpy_to_ros(location, transform.translation)
        # transform.translation.x = location[0]
        # transform.translation.y = location[1]
        # transform.translation.z = location[2]
        ros_utils.quaternion_numpy_to_ros(orientation_quat, transform.rotation)
        # transform.rotation.x = orientation_quat[0]
        # transform.rotation.y = orientation_quat[1]
        # transform.rotation.z = orientation_quat[2]
        # transform.rotation.w = orientation_quat[3]
        return transform

    def _convert_voxels_to_msg(self, voxels):
        voxels_msg = []
        for voxel in voxels:
            voxel_msg = geometry_msgs.msg.Point()
            ros_utils.point_numpy_to_ros(voxel, voxel_msg)
            # voxel_msg.x = voxel[0]
            # voxel_msg.y = voxel[1]
            # voxel_msg.z = voxel[2]
            voxels_msg.append(voxel_msg)
        return voxels_msg

    def _convert_rays_to_msg(self, rays):
        rays_msg = []
        for ray in rays:
            ray_msg = Ray()
            ros_utils.point_numpy_to_ros(ray.origin(), ray_msg.origin)
            # ray_msg.origin.x = ray.origin()[0]
            # ray_msg.origin.y = ray.origin()[1]
            # ray_msg.origin.z = ray.origin()[2]
            ros_utils.point_numpy_to_ros(ray.direction(), ray_msg.direction)
            # ray_msg.direction.x = ray.direction()[0]
            # ray_msg.direction.y = ray.direction()[1]
            # ray_msg.direction.z = ray.direction()[2]
            rays_msg.append(ray_msg)
        return rays_msg

    def _request_set_score_bounding_box(self, score_bbox, verbose=False):
        try:
            request = SetScoreBoundingBoxRequest()
            ros_utils.point_numpy_to_ros(score_bbox.minimum(), request.min)
            ros_utils.point_numpy_to_ros(score_bbox.maximum(), request.max)
            response = self._set_score_bounding_box_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Set score bounding box service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Set score bounding box: {}".format(exc))
        if verbose:
            rospy.loginfo("Set score bounding box took {}s".format(response.elapsed_seconds))
        return response

    def _request_reset(self, verbose=False):
        try:
            request = EmptyRequest()
            response = self._reset_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Reset service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Reset: {}".format(exc))
        if verbose:
            rospy.loginfo("Reset took {}s".format(response.elapsed_seconds))
        return response

    def _request_insert_point_cloud(self, transform_msg, point_cloud_msg, simulate=False, verbose=False):
        try:
            request = InsertPointCloudRequest()
            request.point_cloud = point_cloud_msg
            request.sensor_to_world = transform_msg
            request.simulate = simulate
            response = self._insert_point_cloud_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Point cloud service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Insertion of point cloud: {}".format(exc))
        if verbose:
            rospy.loginfo("Inserting point cloud took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Received score: {}".format(response.score))
            rospy.loginfo("Received reward: {}".format(response.reward))
        return response

    def _request_insert_depth_map(self, transform_msg, depth_image, intrinsics,
                                  downsample_to_grid=False, simulate=False, verbose=False):
        try:
            request = InsertDepthMapRequest()
            request.header.stamp = rospy.Time.now()
            request.header.frame_id = self._world_frame_id
            request.sensor_to_world = transform_msg
            request.height = depth_image.shape[0]
            request.width = depth_image.shape[1]
            request.stride = request.width
            request.depths = np.asarray(depth_image.flatten(), dtype=np.float32)
            request.focal_length_x = intrinsics[0, 0]
            request.focal_length_y = intrinsics[1, 1]
            request.principal_point_x = intrinsics[0, 2]
            request.principal_point_y = intrinsics[1, 2]
            request.downsample_to_grid = downsample_to_grid
            request.simulate = simulate
            response = self._insert_depth_map_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Depth map service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Insertion of depth map: {}".format(exc))
        if verbose:
            rospy.loginfo("Inserting depth map took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Received score: {}".format(response.score))
            rospy.loginfo("Received reward: {}".format(response.reward))
        return response

    def _request_info(self, verbose=False):
        try:
            request = InfoRequest()
            response = self._info_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Info service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Query voxels: {}".format(exc))
        if verbose:
            rospy.loginfo("Info took {}s".format(response.elapsed_seconds))
        return response

    def _request_query_voxels(self, voxels, verbose=False):
        try:
            request = QueryVoxelsRequest()
            request.voxels = voxels
            response = self._query_voxels_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Query voxels service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Query voxels: {}".format(exc))
        if verbose:
            rospy.loginfo("Query voxels took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Number of occupied voxels: {}".format(response.num_occupied))
            rospy.loginfo("Number of free voxels: {}".format(response.num_free))
            rospy.loginfo("Number of unknown voxels: {}".format(response.num_unknown))
            rospy.loginfo("Expected reward: {}".format(response.expected_reward))
        if self._raycast_point_cloud_pub is not None:
          pc = response.point_cloud
          pc.header.stamp = rospy.Time.now()
          pc.header.frame_id = self._world_frame_id
          self._raycast_point_cloud_pub.publish(pc)
        return response

    def _request_query_subvolume(self, request, verbose=False):
        try:
            response = self._query_subvolume_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Query subvolume service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Query subvolume: {}".format(exc))
        if verbose:
            rospy.loginfo("Query subvolume took {}s".format(response.elapsed_seconds))
        return response

    def _request_raycast(self, rays, ignore_unknown_voxels, max_range=-1,
                         only_in_score_bounding_box=False, verbose=False):
        try:
            request = RaycastRequest()
            request.rays = rays
            request.ignore_unknown_voxels = ignore_unknown_voxels
            request.max_range = max_range
            request.only_in_score_bounding_box = only_in_score_bounding_box
            response = self._raycast_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Raycast service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Raycast: {}".format(exc))
        if verbose:
            rospy.loginfo("Raycast took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Number of hit occupied voxels: {}".format(response.num_hits_occupied))
            rospy.loginfo("Number of hit free voxels: {}".format(response.num_hits_free))
            rospy.loginfo("Number of hit unknown voxels: {}".format(response.num_hits_unknown))
            rospy.loginfo("Expected reward: {}".format(response.expected_reward))
            rospy.loginfo("Point cloud size: {}".format(response.point_cloud.width * response.point_cloud.height))
        if self._raycast_point_cloud_pub is not None:
            pc = response.point_cloud
            pc.header.stamp = rospy.Time.now()
            pc.header.frame_id = self._world_frame_id
            self._raycast_point_cloud_pub.publish(pc)
        return response

    def _request_raycast_camera(self, transform_msg, width, height, focal_length,
                                ignore_unknown_voxels, max_range=-1, verbose=False):
        try:
            request = RaycastCameraRequest()
            request.sensor_to_world = transform_msg
            request.height = height
            request.width = width
            request.focal_length = focal_length
            request.ignore_unknown_voxels = ignore_unknown_voxels
            request.max_range = max_range
            response = self._raycast_camera_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Raycast camera service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Raycast camera: {}".format(exc))
        if verbose:
            rospy.loginfo("Raycast camera took {}s".format(response.elapsed_seconds))
            rospy.loginfo("Number of hit occupied voxels: {}".format(response.num_hits_occupied))
            rospy.loginfo("Number of hit free voxels: {}".format(response.num_hits_free))
            rospy.loginfo("Number of hit unknown voxels: {}".format(response.num_hits_unknown))
            rospy.loginfo("Expected reward: {}".format(response.expected_reward))
            rospy.loginfo("Point cloud size: {}".format(response.point_cloud.width * response.point_cloud.height))
        if self._raycast_point_cloud_pub is not None:
          pc = response.point_cloud
          pc.header.stamp = rospy.Time.now()
          pc.header.frame_id = self._world_frame_id
          self._raycast_point_cloud_pub.publish(pc)
        return response

    def _request_clear_bounding_box_voxels(self, bbox, densify):
        try:
            request = ClearBoundingBoxRequest()
            ros_utils.point_numpy_to_ros(bbox.minimum(), request.min)
            ros_utils.point_numpy_to_ros(bbox.maximum(), request.max)
            request.densify = densify
            response = self._clear_bounding_box_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Clear bounding box service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Clear bounding box failed: {}".format(exc))
        assert(response.success)
        return response

    def _request_override_bounding_box_voxels(self, bbox, occupancy, observation_count, densify):
        try:
            request = OverrideBoundingBoxRequest()
            ros_utils.point_numpy_to_ros(bbox.minimum(), request.min)
            ros_utils.point_numpy_to_ros(bbox.maximum(), request.max)
            request.occupancy = occupancy
            request.observation_count = observation_count
            request.densify = densify
            response = self._override_bounding_box_service(request)
        except (rospy.ServiceException, rospy.exceptions.TransportTerminated) as exc:
            rospy.logwarn("WARNING: Override bounding box service did not process request: {}".format(str(exc)))
            rospy.logwarn("WARNING: Trying to reconnect to service")
            self._connect_services()
            raise self.Exception("Override bounding box: {}".format(exc))
        assert(response.success)
        return response

    def perform_insert_point_cloud_rpy(self, location, orientation_rpy, point_cloud, simulate=False):
        orientation_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        return self.perform_insert_point_cloud_quat(location, orientation_quat, point_cloud, simulate)

    def perform_insert_point_cloud_quat(self, location, orientation_quat, point_cloud, simulate=False):
        # timer = Timer()
        sensor_to_world = self._get_transform_msg_quat(location, orientation_quat)
        # t1 = timer.elapsed_seconds()
        point_cloud_msg = self._create_point_cloud_msg(point_cloud)
        # t2 = timer.elapsed_seconds()
        # rospy.logdebug("Requesting point cloud insertion")
        result = self._request_insert_point_cloud(sensor_to_world, point_cloud_msg, simulate)
        # t3 = timer.elapsed_seconds()
        # print("Timing of update_map_quat():")
        # print("  ", t1)
        # print("  ", t2 - t1)
        # print("  ", t3 - t2)
        # print("Total: ", t3)
        return result

    def perform_insert_depth_map_rpy(self, location, orientation_rpy, depth_image, intrinsics,
                                     downsample_to_grid=False, simulate=False):
        orientation_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        return self.perform_insert_depth_map_quat(location, orientation_quat, depth_image, intrinsics,
                                                  downsample_to_grid, simulate)

    def perform_insert_depth_map_quat(self, location, orientation_quat, depth_image, intrinsics,
                                      downsample_to_grid=False, simulate=False):
        # timer = Timer()
        sensor_to_world = self._get_transform_msg_quat(location, orientation_quat)
        # t1 = timer.elapsed_seconds()
        # rospy.logdebug("Requesting depth map insertion")
        result = self._request_insert_depth_map(sensor_to_world, depth_image, intrinsics,
                                                downsample_to_grid, simulate)
        # t3 = timer.elapsed_seconds()
        # print("Timing of update_map_quat():")
        # print("  ", t1)
        # print("  ", t2 - t1)
        # print("  ", t3 - t2)
        # print("Total: ", t3)
        return result

    def perform_set_score_bounding_box(self, score_bbox):
        rospy.logdebug("Requesting set score bounding box")
        return self._request_set_score_bounding_box(score_bbox)

    def perform_reset(self):
        rospy.logdebug("Requesting reset")
        return self._request_reset()

    def perform_info(self):
        rospy.logdebug("Requesting info")
        return self._request_info()

    def perform_query_voxels(self, voxels):
        voxels_msg = self._convert_voxels_to_msg(voxels)
        # Request query voxels
        rospy.logdebug("Requesting query voxels")
        return self._request_query_voxels(voxels_msg)

    def perform_query_subvolume_quat(self, center, orientation_quat, level, size_x, size_y, size_z, axis_mode=0):
        subvolume_msg = QuerySubvolumeRequest()
        ros_utils.point_numpy_to_ros(center, subvolume_msg.center)
        ros_utils.quaternion_numpy_to_ros(orientation_quat, subvolume_msg.orientation)
        subvolume_msg.level = level
        subvolume_msg.size_x = size_x
        subvolume_msg.size_y = size_y
        subvolume_msg.size_z = size_z
        subvolume_msg.axis_mode = axis_mode
        rospy.logdebug("Requesting query subvolume")
        return self._request_query_subvolume(subvolume_msg)

    def perform_query_subvolume_rpy(self, center, orientation_rpy, level, size_x, size_y, size_z, axis_mode=0):
        orientation_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        return self.perform_query_subvolume_quat(center, orientation_quat, level, size_x, size_y, size_z, axis_mode)

    def perform_raycast(self, rays, ignore_unknown_voxels=False, max_range=-1,
                        only_in_score_bounding_box=False, convert_point_cloud=False):
        rays_msg = self._convert_rays_to_msg(rays)
        # Request raycast
        rospy.logdebug("Requesting raycast")
        result = self._request_raycast(rays_msg, ignore_unknown_voxels, max_range, only_in_score_bounding_box)
        if convert_point_cloud:
            result.point_cloud = self.convert_raycast_point_cloud_from_msg(result.point_cloud)
        return result

    def convert_raycast_point_cloud_from_msg(self, point_cloud_msg):
        field_names = ["x", "y", "z", "occupancy", "observation_certainty", "is_surface", "is_known"]
        assert(len(field_names) == len(point_cloud_msg.fields))
        for i, field_name in enumerate(field_names):
            assert(point_cloud_msg.fields[i].name == field_name)
        if point_cloud_msg.is_bigendian:
          assert(sys.byteorder == 'big')
        else:
           assert(sys.byteorder == 'little')
        fmt = "@fffff??"
        fmt_size = struct.calcsize(fmt)
        point_cloud = []
        index = 0
        for y in xrange(point_cloud_msg.height):
            for x in xrange(point_cloud_msg.width):
                data = point_cloud_msg.data[index:index + fmt_size]
                x, y, z, occupancy, observation_certainty, is_surface, is_known = struct.unpack(fmt, data)
                xyz = np.array([x, y, z])
                point_cloud.append(self.RaycastResultPoint(xyz, occupancy, observation_certainty, is_surface, is_known))
                index += point_cloud_msg.point_step
            index += point_cloud_msg.row_step
        return point_cloud

    def perform_raycast_camera_rpy(self, location, orientation_rpy,
                                   width, height, focal_length,
                                   ignore_unknown_voxels=False, max_range=-1):
        orientation_quat = math_utils.convert_rpy_to_quat(orientation_rpy)
        return self.perform_raycast_camera_quat(
          location, orientation_quat, width, height, focal_length, ignore_unknown_voxels, max_range)

    def perform_raycast_camera_quat(self, location, orientation_quat,
                                    width, height, focal_length,
                                    ignore_unknown_voxels=False,
                                    max_range=-1):
        sensor_to_world = self._get_transform_msg_quat(location, orientation_quat)
        # Request raycast
        rospy.logdebug("Requesting raycast camera")
        return self._request_raycast_camera(
          sensor_to_world, width, height, focal_length, ignore_unknown_voxels, max_range)

    def perform_clear_bounding_box_voxels(self, bbox, densify=True):
        rospy.logdebug("Requesting clear bounding box")
        rospy.loginfo("Clearing bounding box: {}".format(bbox))
        return self._request_clear_bounding_box_voxels(bbox, densify)

    def perform_override_bounding_box_voxels(self, bbox, occupancy, observation_count, densify=True):
        rospy.logdebug("Requesting override bounding box")
        rospy.loginfo("Overriding bounding box: {} with {}".format(bbox, occupancy))
        return self._request_override_bounding_box_voxels(bbox, occupancy, observation_count, densify)
