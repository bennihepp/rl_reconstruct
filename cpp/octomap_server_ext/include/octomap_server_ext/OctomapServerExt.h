/*
 * Copyright (c) 2010-2013, A. Hornung, University of Freiburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OCTOMAP_SERVER_OCTOMAPSERVER_EXT_H
#define OCTOMAP_SERVER_OCTOMAPSERVER_EXT_H

#include <unordered_set>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/ColorRGBA.h>

// #include <moveit_msgs/CollisionObject.h>
// #include <moveit_msgs/CollisionMap.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_srvs/Empty.h>
#include <dynamic_reconfigure/server.h>
#include <octomap_server_ext/OctomapServerExtConfig.h>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>


#include <tf/transform_listener.h>
#include <tf/message_filter.h>
#include <message_filters/subscriber.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_server_ext/ClearBoundingBox.h>
#include <octomap_server_ext/OverrideBoundingBox.h>
#include <octomap_server_ext/InsertPointCloud.h>
#include <octomap_server_ext/QueryVoxels.h>
#include <octomap_server_ext/Raycast.h>
#include <octomap_server_ext/RaycastCamera.h>

#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>

//#define COLOR_OCTOMAP_SERVER // turned off here, turned on identical ColorOctomapServer.h - easier maintenance, only maintain OctomapServer and then copy and paste to ColorOctomapServer and change define. There are prettier ways to do this, but this works for now

#ifdef COLOR_OCTOMAP_SERVER
#include <octomap/ColorOcTree.h>
#endif

namespace octomap_server_ext {
class OctomapServerExt {

public:
#ifdef COLOR_OCTOMAP_SERVER
  typedef pcl::PointXYZRGB PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZRGB> PCLPointCloud;
  typedef octomap::ColorOcTree OcTreeT;
#else
  typedef pcl::PointXYZ PCLPoint;
  typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
  typedef octomap::OcTree OcTreeT;
#endif
  typedef octomap_msgs::GetOctomap OctomapSrv;

  struct PointXYZExt {
    union {
      float xyz[3];
      struct {
        float x;
        float y;
        float z;
      };
    };
    float occupancy;
    bool is_surface;
    bool is_known;
#ifdef COLOR_OCTOMAP_SERVER
    union {
      uint_8 rgb[3];
      struct {
        uint_8 r;
        uint_8 g;
        uint_8 b;
      }
    };
#endif
  };
  using PointCloudExt = pcl::PointCloud<PointXYZExt>;

  struct Ray {
      octomath::Vector3 origin;
      octomath::Vector3 direction;
  };

  struct QueryVoxelsResult {
      std::size_t num_occupied;
      std::size_t num_free;
      std::size_t num_unknown;
      double expected_reward;
      pcl::PointCloud<PointXYZExt> point_cloud;
  };

  struct RaycastResult {
      std::size_t num_hits_occupied;
      std::size_t num_hits_free;
      std::size_t num_hits_unknown;
      double expected_reward;
      pcl::PointCloud<PointXYZExt> point_cloud;
  };

  OctomapServerExt(ros::NodeHandle private_nh_ = ros::NodeHandle("~"));
  virtual ~OctomapServerExt();
  virtual bool octomapBinarySrv(OctomapSrv::Request  &req, OctomapSrv::GetOctomap::Response &res);
  virtual bool octomapFullSrv(OctomapSrv::Request  &req, OctomapSrv::GetOctomap::Response &res);
  bool resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);

  void filterHeightPointCloud(PCLPointCloud& cloud);
  virtual bool clearBoundingBoxSrv(ClearBoundingBox::Request& req, ClearBoundingBox::Response& resp);
  virtual bool overrideBoundingBoxSrv(OverrideBoundingBox::Request &req, OverrideBoundingBox::Response &res);
  virtual void insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
  virtual bool insertPointCloudSrv(InsertPointCloud::Request &req, InsertPointCloud::Response &res);
  virtual bool queryVoxelsSrv(QueryVoxels::Request &req, QueryVoxels::Response &res);
  virtual bool raycastSrv(Raycast::Request &req, Raycast::Response &res);
  virtual bool raycastCameraSrv(RaycastCamera::Request &req, RaycastCamera::Response &res);
  void pointCloudExtToROSMsg(const PointCloudExt& pcl_cloud, sensor_msgs::PointCloud2& ros_cloud);
  virtual bool openFile(const std::string& filename);

  void readSurfaceVoxels(const std::string& filename);
  double computeScore() const;

protected:
  inline static void updateMinKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& min) {
    for (unsigned i = 0; i < 3; ++i)
      min[i] = std::min(in[i], min[i]);
  };

  inline static void updateMaxKey(const octomap::OcTreeKey& in, octomap::OcTreeKey& max) {
    for (unsigned i = 0; i < 3; ++i)
      max[i] = std::max(in[i], max[i]);
  };

  /// Test if key is within update area of map (2D, ignores height)
  inline bool isInUpdateBBX(const OcTreeT::iterator& it) const {
    // 2^(tree_depth-depth) voxels wide:
    unsigned voxelWidth = (1 << (m_maxTreeDepth - it.getDepth()));
    octomap::OcTreeKey key = it.getIndexKey(); // lower corner of voxel
    return (key[0] + voxelWidth >= m_updateBBXMin[0]
            && key[1] + voxelWidth >= m_updateBBXMin[1]
            && key[0] <= m_updateBBXMax[0]
            && key[1] <= m_updateBBXMax[1]);
  }

  void reconfigureCallback(octomap_server_ext::OctomapServerExtConfig& config, uint32_t level);
  void publishBinaryOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  void publishFullOctoMap(const ros::Time& rostime = ros::Time::now()) const;
  virtual void publishAll(const ros::Time& rostime = ros::Time::now());

  /**
  * @brief override occupancy of a bounding box volume
  *
  * @param min Minimum of bounding box
  * @param max Maximum of bounding box
  * @param occupancy Occupancy value to override
  * @param densify Whether to create new nodes in unknown space
  */
  virtual void overrideBoundingBox(const octomath::Vector3& min, const octomath::Vector3& max,
                                   const double occupancy, const bool densify);

  /**
  * @brief perform a query of voxels on the occupancy map
  *
  * @param voxels The voxels to query
  * @return Results of the query
  */
  virtual QueryVoxelsResult queryVoxels(const std::vector<octomath::Vector3>& voxels);

  /**
  * @brief perform raycast of rays
  *
  * @param rays The rays to cast
  * @param ignore_unknown_voxels Whether to ignore unknown voxels when casting the ray
  * @return Results of the raycast including Point cloud of hit occupied voxels
  */
  virtual RaycastResult raycast(const std::vector<Ray>& rays,
                                const bool ignore_unknown_voxels = false,
                                const float max_range = -1);

  /**
  * @brief perform raycast for pinhole camera
  * The returned point cloud is in global map frame.
  *
  * @param sensor_to_world Transform from sensor to world frame
  * @param height Height of camera image plane
  * @param width Height of camera image plane
  * @param focal_length Focal length of pinhole camera
  * @return Results of the raycast including Point cloud of hit occupied voxels
  */
  virtual RaycastResult raycastCamera(const tf::Transform& sensor_to_world_tf,
                                      const int height, const int width,
                                      const float focal_length,
                                      const bool ignore_unknown_voxels = false,
                                      const float max_range = -1);

  /**
  * @brief update occupancy map with a scan
  * The scans should be in the global map frame.
  *
  * @param sensorOrigin origin of the measurements for raycasting
  * @param pc scan endpoints
  */
  virtual void insertScan(const tf::Point& sensorOrigin, const PCLPointCloud& pc);

  /**
  * @brief update occupancy map with a scan labeled as ground and nonground.
  * The scans should be in the global map frame.
  *
  * @param sensorOrigin origin of the measurements for raycasting
  * @param ground scan endpoints on the ground plane (only clear space)
  * @param nonground all other endpoints (clear up to occupied endpoint)
  */
  virtual void insertScan(const tf::Point& sensorOrigin, const PCLPointCloud& ground, const PCLPointCloud& nonground);

  /// label the input cloud "pc" into ground and nonground. Should be in the robot's fixed frame (not world!)
  void filterGroundPlane(const PCLPointCloud& pc, PCLPointCloud& ground, PCLPointCloud& nonground) const;

  /**
  * @brief Find speckle nodes (single occupied voxels with no neighbors). Only works on lowest resolution!
  * @param key
  * @return
  */
  bool isSpeckleNode(const octomap::OcTreeKey& key) const;

  /// hook that is called before traversing all nodes
  virtual void handlePreNodeTraversal(const ros::Time& rostime);

  /// hook that is called when traversing all nodes of the updated Octree (does nothing here)
  virtual void handleNode(const OcTreeT::iterator& it) {};

  /// hook that is called when traversing all nodes of the updated Octree in the updated area (does nothing here)
  virtual void handleNodeInBBX(const OcTreeT::iterator& it) {};

  /// hook that is called when traversing occupied nodes of the updated Octree
  virtual void handleOccupiedNode(const OcTreeT::iterator& it);

  /// hook that is called when traversing occupied nodes in the updated area (updates 2D map projection here)
  virtual void handleOccupiedNodeInBBX(const OcTreeT::iterator& it);

  /// hook that is called when traversing free nodes of the updated Octree
  virtual void handleFreeNode(const OcTreeT::iterator& it);

  /// hook that is called when traversing free nodes in the updated area (updates 2D map projection here)
  virtual void handleFreeNodeInBBX(const OcTreeT::iterator& it);

  /// hook that is called after traversing all nodes
  virtual void handlePostNodeTraversal(const ros::Time& rostime);

  /// updates the downprojected 2D map as either occupied or free
  virtual void update2DMap(const OcTreeT::iterator& it, bool occupied);

  inline unsigned mapIdx(int i, int j) const {
    return m_gridmap.info.width * j + i;
  }

  inline unsigned mapIdx(const octomap::OcTreeKey& key) const {
    return mapIdx((key[0] - m_paddedMinKey[0]) / m_multires2DScale,
                  (key[1] - m_paddedMinKey[1]) / m_multires2DScale);

  }

  /**
   * Adjust data of map due to a change in its info properties (origin or size,
   * resolution needs to stay fixed). map already contains the new map info,
   * but the data is stored according to oldMapInfo.
   */

  void adjustMapData(nav_msgs::OccupancyGrid& map, const nav_msgs::MapMetaData& oldMapInfo) const;

  inline bool mapChanged(const nav_msgs::MapMetaData& oldMapInfo, const nav_msgs::MapMetaData& newMapInfo) {
    return (    oldMapInfo.height != newMapInfo.height
                || oldMapInfo.width != newMapInfo.width
                || oldMapInfo.origin.position.x != newMapInfo.origin.position.x
                || oldMapInfo.origin.position.y != newMapInfo.origin.position.y);
  }

  static std_msgs::ColorRGBA heightMapColor(double h);
  ros::NodeHandle m_nh;
  ros::Publisher  m_markerPub, m_binaryMapPub, m_fullMapPub, m_pointCloudPub, m_collisionObjectPub, m_mapPub, m_cmapPub, m_fmapPub, m_fmarkerPub;
  message_filters::Subscriber<sensor_msgs::PointCloud2>* m_pointCloudSub;
  tf::MessageFilter<sensor_msgs::PointCloud2>* m_tfPointCloudSub;
  ros::ServiceServer m_clearBoundingBoxService;
  ros::ServiceServer m_overrideBoundingBoxService;
  ros::ServiceServer m_insertPointCloudService;
  ros::ServiceServer m_queryVoxelsService;
  ros::ServiceServer m_raycastService;
  ros::ServiceServer m_raycastCameraService;
  ros::ServiceServer m_octomapBinaryService, m_octomapFullService, m_resetService;
  tf::TransformListener m_tfListener;
  boost::recursive_mutex m_config_mutex;
  dynamic_reconfigure::Server<OctomapServerExtConfig> m_reconfigureServer;

  OcTreeT* m_octree;
  octomap::KeyRay m_keyRay;  // temp storage for ray casting
  octomap::OcTreeKey m_updateBBXMin;
  octomap::OcTreeKey m_updateBBXMax;

  double m_maxRange;
  std::string m_worldFrameId; // the map frame
  std::string m_baseFrameId; // base of the robot for ground plane filtering
  bool m_useHeightMap;
  std_msgs::ColorRGBA m_color;
  std_msgs::ColorRGBA m_colorFree;
  double m_colorFactor;

  bool m_latchedTopics;
  bool m_publishFreeSpace;

  double m_res;
  unsigned m_treeDepth;
  unsigned m_maxTreeDepth;

  double m_pointcloudMinX;
  double m_pointcloudMaxX;
  double m_pointcloudMinY;
  double m_pointcloudMaxY;
  double m_pointcloudMinZ;
  double m_pointcloudMaxZ;
  double m_occupancyMinZ;
  double m_occupancyMaxZ;
  double m_minSizeX;
  double m_minSizeY;
  bool m_filterSpeckles;

  bool m_filterGroundPlane;
  double m_groundFilterDistance;
  double m_groundFilterAngle;
  double m_groundFilterPlaneDistance;

  bool m_compressMap;

  bool m_initConfig;

  // downprojected 2D map:
  bool m_incrementalUpdate;
  nav_msgs::OccupancyGrid m_gridmap;
  bool m_publish2DMap;
  bool m_mapOriginChanged;
  octomap::OcTreeKey m_paddedMinKey;
  unsigned m_multires2DScale;
  bool m_projectCompleteMap;
  bool m_useColoredMap;

  std::string m_surfaceVoxelsFilename;
  std::vector<octomath::Vector3> m_surfaceVoxels;
  std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> m_surfaceVoxelKeys;
  float m_voxelFreeThreshold;
  float m_voxelOccupiedThreshold;
  double m_scorePerVoxel;
  double m_scorePerSurfaceVoxel;
  bool m_useOnlySurfaceVoxelsForScore;
  double m_score;
};
}

#endif
