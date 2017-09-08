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

#include <octomap_server_ext/OctomapServerExt.h>
#include <iostream>
#include <sstream>
#include <bh/common.h>
#include <octomap_ext/OcTreeKey.h>
#include <pcl/filters/voxel_grid.h>
#include <bh/math/geometry.h>

using namespace octomap;
using octomap_msgs::Octomap;

bool is_equal (double a, double b, double epsilon = 1.0e-7)
{
    return std::abs(a - b) < epsilon;
}

namespace octomap_server_ext {

OctomapServerExt::OctomapServerExt(ros::NodeHandle private_nh_)
: m_nh(),
  m_pointCloudSub(NULL),
  m_tfPointCloudSub(NULL),
  m_reconfigureServer(m_config_mutex),
  m_octree(NULL),
  m_maxRange(-1.0),
  m_worldFrameId("/map"), m_baseFrameId("base_footprint"),
  m_useHeightMap(true),
  m_useColoredMap(false),
  m_colorFactor(0.8),
  m_latchedTopics(true),
  m_publishFreeSpace(false),
  m_res(0.05),
  m_treeDepth(0),
  m_maxTreeDepth(0),
  m_pointcloudMinX(-std::numeric_limits<double>::max()),
  m_pointcloudMaxX(std::numeric_limits<double>::max()),
  m_pointcloudMinY(-std::numeric_limits<double>::max()),
  m_pointcloudMaxY(std::numeric_limits<double>::max()),
  m_pointcloudMinZ(-std::numeric_limits<double>::max()),
  m_pointcloudMaxZ(std::numeric_limits<double>::max()),
  m_occupancyMinZ(-std::numeric_limits<double>::max()),
  m_occupancyMaxZ(std::numeric_limits<double>::max()),
  m_minSizeX(0.0), m_minSizeY(0.0),
  m_filterSpeckles(false), m_filterGroundPlane(false),
  m_groundFilterDistance(0.04), m_groundFilterAngle(0.15), m_groundFilterPlaneDistance(0.07),
  m_compressMap(true),
  m_incrementalUpdate(false),
  m_initConfig(true),
  m_useOnlySurfaceVoxelsForScore(true),
  m_score(0.0),
  m_normalized_score(0.0),
  m_probabilistic_score(0.0),
  m_normalized_probabilistic_score(0.0)
{
  const float lowest = std::numeric_limits<float>::lowest();
  const float highest = std::numeric_limits<float>::max();
  m_scoreBoundingBoxMin = point3d(lowest, lowest, lowest);
  m_scoreBoundingBoxMax = point3d(highest, highest, highest);
  m_scoreBoundingBox = bh::BoundingBox3Df(octomapToBh(m_scoreBoundingBoxMin), octomapToBh(m_scoreBoundingBoxMax));

  double probHit, probMiss, thresMin, thresMax;

  ros::NodeHandle private_nh(private_nh_);
  private_nh.param("frame_id", m_worldFrameId, m_worldFrameId);
  private_nh.param("base_frame_id", m_baseFrameId, m_baseFrameId);
  private_nh.param("height_map", m_useHeightMap, m_useHeightMap);
  private_nh.param("colored_map", m_useColoredMap, m_useColoredMap);
  private_nh.param("color_factor", m_colorFactor, m_colorFactor);

  private_nh.param("pointcloud_min_x", m_pointcloudMinX,m_pointcloudMinX);
  private_nh.param("pointcloud_max_x", m_pointcloudMaxX,m_pointcloudMaxX);
  private_nh.param("pointcloud_min_y", m_pointcloudMinY,m_pointcloudMinY);
  private_nh.param("pointcloud_max_y", m_pointcloudMaxY,m_pointcloudMaxY);
  private_nh.param("pointcloud_min_z", m_pointcloudMinZ,m_pointcloudMinZ);
  private_nh.param("pointcloud_max_z", m_pointcloudMaxZ,m_pointcloudMaxZ);
  private_nh.param("occupancy_min_z", m_occupancyMinZ,m_occupancyMinZ);
  private_nh.param("occupancy_max_z", m_occupancyMaxZ,m_occupancyMaxZ);
  private_nh.param("min_x_size", m_minSizeX,m_minSizeX);
  private_nh.param("min_y_size", m_minSizeY,m_minSizeY);

  private_nh.param("filter_speckles", m_filterSpeckles, m_filterSpeckles);
  private_nh.param("filter_ground", m_filterGroundPlane, m_filterGroundPlane);
  // distance of points from plane for RANSAC
  private_nh.param("ground_filter/distance", m_groundFilterDistance, m_groundFilterDistance);
  // angular derivation of found plane:
  private_nh.param("ground_filter/angle", m_groundFilterAngle, m_groundFilterAngle);
  // distance of found plane from z=0 to be detected as ground (e.g. to exclude tables)
  private_nh.param("ground_filter/plane_distance", m_groundFilterPlaneDistance, m_groundFilterPlaneDistance);

  private_nh.param("sensor_model/max_range", m_maxRange, m_maxRange);

  private_nh.param("resolution", m_res, m_res);
  private_nh.param("sensor_model/hit", probHit, 0.7);
  private_nh.param("sensor_model/miss", probMiss, 0.4);
  private_nh.param("sensor_model/min", thresMin, 0.12);
  private_nh.param("sensor_model/max", thresMax, 0.97);
  private_nh.param("compress_map", m_compressMap, m_compressMap);
  private_nh.param("incremental_2D_projection", m_incrementalUpdate, m_incrementalUpdate);

  if (m_filterGroundPlane && (m_pointcloudMinZ > 0.0 || m_pointcloudMaxZ < 0.0)){
    ROS_WARN_STREAM("You enabled ground filtering but incoming pointclouds will be pre-filtered in ["
              <<m_pointcloudMinZ <<", "<< m_pointcloudMaxZ << "], excluding the ground level z=0. "
              << "This will not work.");
  }

  if (m_useHeightMap && m_useColoredMap) {
    ROS_WARN_STREAM("You enabled both height map and RGB color registration. This is contradictory. Defaulting to height map.");
    m_useColoredMap = false;
  }

  if (m_useColoredMap) {
#ifdef COLOR_OCTOMAP_SERVER
    ROS_INFO_STREAM("Using RGB color registration (if information available)");
#else
    ROS_ERROR_STREAM("Colored map requested in launch file - node not running/compiled to support colors, please define COLOR_OCTOMAP_SERVER and recompile or launch the octomap_color_server node");
#endif
  }


  // initialize octomap object & params
  m_octree = new OcTreeT(m_res);
  m_octree->setProbHit(probHit);
  m_octree->setProbMiss(probMiss);
  m_octree->setClampingThresMin(thresMin);
  m_octree->setClampingThresMax(thresMax);
  m_treeDepth = m_octree->getTreeDepth();
  m_maxTreeDepth = m_treeDepth;
  m_gridmap.info.resolution = m_res;

  double r, g, b, a;
  private_nh.param("color/r", r, 0.0);
  private_nh.param("color/g", g, 0.0);
  private_nh.param("color/b", b, 1.0);
  private_nh.param("color/a", a, 1.0);
  m_color.r = r;
  m_color.g = g;
  m_color.b = b;
  m_color.a = a;

  private_nh.param("color_free/r", r, 0.0);
  private_nh.param("color_free/g", g, 1.0);
  private_nh.param("color_free/b", b, 0.0);
  private_nh.param("color_free/a", a, 1.0);
  m_colorFree.r = r;
  m_colorFree.g = g;
  m_colorFree.b = b;
  m_colorFree.a = a;

  private_nh.param("publish_free_space", m_publishFreeSpace, m_publishFreeSpace);

  private_nh.param("latch", m_latchedTopics, m_latchedTopics);
  if (m_latchedTopics){
    ROS_INFO("Publishing latched (single publish will take longer, all topics are prepared)");
  } else {
    ROS_INFO("Publishing non-latched (topics are only prepared as needed, will only be re-published on map change");
  }

  private_nh.param("use_only_surface_voxels_for_score", m_useOnlySurfaceVoxelsForScore, m_useOnlySurfaceVoxelsForScore);
  private_nh.param("binary_surface_voxels_filename", m_binarySurfaceVoxelsFilename, m_binarySurfaceVoxelsFilename);
  private_nh.param("surface_voxel_filename", m_surfaceVoxelsFilename, m_surfaceVoxelsFilename);
  if (m_surfaceVoxelsFilename.empty() && m_binarySurfaceVoxelsFilename.empty()) {
    ROS_WARN("No surface voxel file specified");
    m_useOnlySurfaceVoxelsForScore = false;
  }
  else if (!m_surfaceVoxelsFilename.empty() && !m_binarySurfaceVoxelsFilename.empty()) {
    ROS_WARN("Text and binary surface voxel file specified. Will use binary.");
    readBinarySurfaceVoxels(m_surfaceVoxelsFilename);
  }
  else if (!m_binarySurfaceVoxelsFilename.empty()) {
    ROS_INFO_STREAM("binary_surface_voxels_filename: " << m_binarySurfaceVoxelsFilename);
    readBinarySurfaceVoxels(m_binarySurfaceVoxelsFilename);
  }
  else {
    ROS_INFO_STREAM("surface_voxel_filename: " << m_surfaceVoxelsFilename);
    readSurfaceVoxels(m_surfaceVoxelsFilename);
  }

  m_voxelFreeThreshold = 0.25f;
  m_voxelOccupiedThreshold = 0.75f;
  m_scorePerVoxel = 0.1;
  m_scorePerSurfaceVoxel = 1.0;
  private_nh.param("voxel_free_threshold", m_voxelFreeThreshold, m_voxelFreeThreshold);
  private_nh.param("voxel_occupied_threshold", m_voxelOccupiedThreshold, m_voxelOccupiedThreshold);
  private_nh.param("score_per_voxel", m_scorePerVoxel, m_scorePerVoxel);
  private_nh.param("score_per_surface_voxel", m_scorePerSurfaceVoxel, m_scorePerSurfaceVoxel);
  private_nh.param("observation_count_saturation", m_observation_count_saturation, m_observation_count_saturation);
  ROS_INFO_STREAM("m_voxelOccupiedThreshold=" << m_voxelOccupiedThreshold);

  float spherical_grid_yaw_step_degrees = 1;
  float spherical_grid_pitch_step_degrees = 1;
  private_nh.param("spherical_grid_yaw_step_degrees", spherical_grid_yaw_step_degrees, spherical_grid_yaw_step_degrees);
  private_nh.param("spherical_grid_pitch_step_degrees", spherical_grid_pitch_step_degrees, spherical_grid_pitch_step_degrees);
  ROS_INFO_STREAM("spherical_grid_yaw_step_degrees=" << spherical_grid_yaw_step_degrees);
  ROS_INFO_STREAM("spherical_grid_pitch_step_degrees=" << spherical_grid_pitch_step_degrees);
  m_spherical_grid_yaw_step = spherical_grid_yaw_step_degrees * M_PI / 180.0f;
  m_spherical_grid_pitch_step = spherical_grid_pitch_step_degrees * M_PI / 180.0f;

  m_markerPub = m_nh.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1, m_latchedTopics);
  m_binaryMapPub = m_nh.advertise<Octomap>("octomap_binary", 1, m_latchedTopics);
  m_fullMapPub = m_nh.advertise<Octomap>("octomap_full", 1, m_latchedTopics);
  m_pointCloudPub = m_nh.advertise<sensor_msgs::PointCloud2>("octomap_point_cloud_centers", 1, m_latchedTopics);
  m_mapPub = m_nh.advertise<nav_msgs::OccupancyGrid>("projected_map", 5, m_latchedTopics);
  m_fmarkerPub = m_nh.advertise<visualization_msgs::MarkerArray>("free_cells_vis_array", 1, m_latchedTopics);

  m_subvolumePointCloudPub = m_nh.advertise<sensor_msgs::PointCloud2>("subvolume_point_cloud", 1, m_latchedTopics);

  m_pointCloudSub = new message_filters::Subscriber<sensor_msgs::PointCloud2> (m_nh, "cloud_in", 5);
  m_tfPointCloudSub = new tf::MessageFilter<sensor_msgs::PointCloud2> (*m_pointCloudSub, m_tfListener, m_worldFrameId, 5);
  m_tfPointCloudSub->registerCallback(boost::bind(&OctomapServerExt::insertCloudCallback, this, _1));

  m_infoService = private_nh.advertiseService("info", &OctomapServerExt::infoSrv, this);
  m_resetService = private_nh.advertiseService("reset", &OctomapServerExt::resetSrv, this);
  m_setScoreBoundingBoxService = private_nh.advertiseService("set_score_bounding_box", &OctomapServerExt::setScoreBoundingBoxSrv, this);
  m_clearBoundingBoxService = private_nh.advertiseService("clear_bounding_box", &OctomapServerExt::clearBoundingBoxSrv, this);
  m_overrideBoundingBoxService = private_nh.advertiseService("override_bounding_box", &OctomapServerExt::overrideBoundingBoxSrv, this);
  m_insertDepthMapService = private_nh.advertiseService("insert_depth_map", &OctomapServerExt::insertDepthMapSrv, this);
  m_insertPointCloudService = private_nh.advertiseService("insert_point_cloud", &OctomapServerExt::insertPointCloudSrv, this);
  m_queryVoxelsService = private_nh.advertiseService("query_voxels", &OctomapServerExt::queryVoxelsSrv, this);
  m_querySubvolumeService = private_nh.advertiseService("query_subvolume", &OctomapServerExt::querySubvolumeSrv, this);
  m_raycastService = private_nh.advertiseService("raycast", &OctomapServerExt::raycastSrv, this);
  m_raycastCameraService = private_nh.advertiseService("raycast_camera", &OctomapServerExt::raycastCameraSrv, this);

  m_octomapBinaryService = m_nh.advertiseService("octomap_binary", &OctomapServerExt::octomapBinarySrv, this);
  m_octomapFullService = m_nh.advertiseService("octomap_full", &OctomapServerExt::octomapFullSrv, this);

  dynamic_reconfigure::Server<OctomapServerExtConfig>::CallbackType f;
  f = boost::bind(&OctomapServerExt::reconfigureCallback, this, _1, _2);
  m_reconfigureServer.setCallback(f);
}

OctomapServerExt::~OctomapServerExt(){
  if (m_tfPointCloudSub){
    delete m_tfPointCloudSub;
    m_tfPointCloudSub = NULL;
  }

  if (m_pointCloudSub){
    delete m_pointCloudSub;
    m_pointCloudSub = NULL;
  }


  if (m_octree){
    delete m_octree;
    m_octree = NULL;
  }

}

bool OctomapServerExt::openFile(const std::string& filename){
  if (filename.length() <= 3)
    return false;

  std::string suffix = filename.substr(filename.length()-3, 3);
  if (suffix== ".bt"){
    if (!m_octree->readBinary(filename)){
      return false;
    }
  } else if (suffix == ".ot"){
    AbstractOcTree* tree = AbstractOcTree::read(filename);
    if (!tree){
      return false;
    }
    if (m_octree){
      delete m_octree;
      m_octree = NULL;
    }
    m_octree = dynamic_cast<OcTreeT*>(tree);
    if (!m_octree){
      ROS_ERROR("Could not read OcTree in file, currently there are no other types supported in .ot");
      return false;
    }

  } else{
    return false;
  }

  ROS_INFO("Octomap file %s loaded (%zu nodes).", filename.c_str(),m_octree->size());

  m_treeDepth = m_octree->getTreeDepth();
  m_maxTreeDepth = m_treeDepth;
  m_res = m_octree->getResolution();
  m_gridmap.info.resolution = m_res;
  double minX, minY, minZ;
  double maxX, maxY, maxZ;
  m_octree->getMetricMin(minX, minY, minZ);
  m_octree->getMetricMax(maxX, maxY, maxZ);

  m_updateBBXMin[0] = m_octree->coordToKey(minX);
  m_updateBBXMin[1] = m_octree->coordToKey(minY);
  m_updateBBXMin[2] = m_octree->coordToKey(minZ);

  m_updateBBXMax[0] = m_octree->coordToKey(maxX);
  m_updateBBXMax[1] = m_octree->coordToKey(maxY);
  m_updateBBXMax[2] = m_octree->coordToKey(maxZ);

  const float lowest = std::numeric_limits<float>::lowest();
  const float highest = std::numeric_limits<float>::max();
  m_scoreBoundingBoxMin = point3d(lowest, lowest, lowest);
  m_scoreBoundingBoxMax = point3d(highest, highest, highest);
  m_scoreBoundingBox = bh::BoundingBox3Df(octomapToBh(m_scoreBoundingBoxMin), octomapToBh(m_scoreBoundingBoxMax));

  readSurfaceVoxels(m_surfaceVoxelsFilename);
  std::tie(m_score, m_normalized_score, m_probabilistic_score, m_normalized_probabilistic_score) = computeScore(m_octree);

  publishAll();

  return true;

}

void OctomapServerExt::readSurfaceVoxels(const std::string& filename) {
  m_surfaceVoxels.clear();
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("ERROR: Unable to open surface voxel file: " + filename);
  }
  std::string line_str;
  while (std::getline(in, line_str)) {
    if (line_str.empty()) {
      continue;
    }
    std::istringstream in_str(line_str);
    float x, y, z;
    in_str >> x >> y >> z;
    octomath::Vector3 voxel(x, y, z);
    m_surfaceVoxels.push_back(voxel);
//    ROS_DEBUG_STREAM("Surface voxel: " << voxel(0) << ", " << voxel(1) << ", " << voxel(2));
  }
  in.close();
  m_surfaceVoxelKeys.clear();
  std::transform(m_surfaceVoxels.begin(), m_surfaceVoxels.end(),
                 std::inserter(m_surfaceVoxelKeys, m_surfaceVoxelKeys.end()),
                 [&](const octomath::Vector3& voxel) {
                     const OcTreeKey key = m_octree->coordToKey(voxel);
                     return key;
                 });
  ROS_INFO_STREAM("Read " << m_surfaceVoxels.size() << " surface voxels");
}

void OctomapServerExt::readBinarySurfaceVoxels(const std::string& filename) {
  m_surfaceVoxels.clear();
  std::ifstream in(filename, std::ios_base::binary);
  if (!in) {
    throw std::runtime_error("ERROR: Unable to open binary surface voxel file: " + filename);
  }
  size_t num_voxels;
  in.read(reinterpret_cast<char*>(&num_voxels), sizeof(num_voxels));
  ROS_INFO_STREAM("Reading " << num_voxels << " voxels from binary file");
  for (size_t i = 0; i < num_voxels; ++i) {
    float x, y, z;
    in.read(reinterpret_cast<char*>(&x), sizeof(x));
    in.read(reinterpret_cast<char*>(&y), sizeof(y));
    in.read(reinterpret_cast<char*>(&z), sizeof(z));
    octomath::Vector3 voxel(x, y, z);
    m_surfaceVoxels.push_back(voxel);
  }
  in.close();
  m_surfaceVoxelKeys.clear();
  std::transform(m_surfaceVoxels.begin(), m_surfaceVoxels.end(),
                 std::inserter(m_surfaceVoxelKeys, m_surfaceVoxelKeys.end()),
                 [&](const octomath::Vector3& voxel) {
                     const OcTreeKey key = m_octree->coordToKey(voxel);
                     return key;
                 });
  ROS_INFO_STREAM("Read " << m_surfaceVoxels.size() << " surface voxels");
}

bool OctomapServerExt::isPointInScoreBoundingBox(const point3d& point) const {
  if (point(0) < m_scoreBoundingBoxMin(0)
      || point(1) < m_scoreBoundingBoxMin(1)
      || point(2) < m_scoreBoundingBoxMin(2)) {
    return false;
  }
  if (point(0) > m_scoreBoundingBoxMax(0)
      || point(1) > m_scoreBoundingBoxMax(1)
      || point(2) > m_scoreBoundingBoxMax(2)) {
    return false;
  }
  return true;
}

double OctomapServerExt::computeObservationCertainty(const OcTreeNodeT* node) const {
  const double certainty = (1 - std::exp(-node->getObservationCount() / m_observation_count_saturation));
  return certainty;
}

double OctomapServerExt::computeObservationCertainty(const double observation_count) const {
  const double certainty = (1 - std::exp(-observation_count / m_observation_count_saturation));
  return certainty;
}

std::tuple<double, double, double, double> OctomapServerExt::computeScore2(const OcTreeT* octree) const {
  double score = 0;
  double normalized_score = 0;
  double probabilistic_score = 0;
  double normalized_probabilistic_score = 0;
  std::size_t occupied_count = 0;
  std::size_t free_count = 0;

  if (m_useOnlySurfaceVoxelsForScore) {
    double max_score = 0.0;
    for (const OcTreeKey &key : m_surfaceVoxelKeys) {
      const point3d point = octree->keyToCoord(key);
      if (!isPointInScoreBoundingBox(point)) {
        continue;
      }
      max_score += m_scorePerSurfaceVoxel;
      const OcTreeNodeT *node = octree->search(key);
      float occupancy;
      float observation_count;
      if (node == nullptr) {
        occupancy = 0.5f;
        observation_count = 0;
      }
      else {
        occupancy = node->getOccupancy();
        observation_count = node->getObservationCount();
      }
      double voxel_score;
      double probabilistic_voxel_score;
      std::tie(voxel_score, probabilistic_voxel_score) = computeVoxelScore(occupancy, observation_count, m_scorePerSurfaceVoxel);
      score += voxel_score;
      probabilistic_score += probabilistic_voxel_score;
      // Is voxel already known with certainty?
      if (node != nullptr) {
        if (node->getOccupancy() >= m_voxelOccupiedThreshold || node->getOccupancy() <= m_voxelFreeThreshold) {
          // Update counts
          if (node->getOccupancy() >= m_voxelOccupiedThreshold) {
            ++occupied_count;
          } else {
            ++free_count;
          }
        }
      }
    }
    // Normalize score by total number of surface voxels and score per surface voxel (i.e. between 0 and 1)
    normalized_score = score / max_score;
    normalized_probabilistic_score = probabilistic_score / max_score;
  }
  else {
    // Alternative computation (considering all voxels)
    double max_score = 0.0;
    for (auto it = octree->begin_leafs(); it != octree->end_leafs(); ++it) {
      if (!isPointInScoreBoundingBox(it.getCoordinate())) {
        continue;
      }
      const bool is_surface_voxel = m_surfaceVoxelKeys.find(it.getKey()) != m_surfaceVoxelKeys.end();
      if (is_surface_voxel) {
        max_score += m_scorePerSurfaceVoxel;
      }

      double voxel_score;
      double probabilistic_voxel_score;
      std::tie(voxel_score, probabilistic_voxel_score) = computeVoxelScore(
              it->getOccupancy(), it->getObservationCount(), is_surface_voxel);
      score += voxel_score;
      probabilistic_score += probabilistic_voxel_score;

      // Is voxel already known with certainty?
      if (it->getOccupancy() >= m_voxelOccupiedThreshold || it->getOccupancy() <= m_voxelFreeThreshold) {
        // Update counts
        if (it->getOccupancy() >= m_voxelOccupiedThreshold) {
          ++occupied_count;
        }
        else {
          ++free_count;
        }
      }
    }
    // Normalized score makes not too much sense in this context. Can be higher than 1.
    normalized_score = score / max_score;
    normalized_probabilistic_score = probabilistic_score / max_score;
  }
  ROS_DEBUG_STREAM("occupied voxels: " << occupied_count << ", free voxels: " << free_count);
  ROS_DEBUG_STREAM("score is " << score << ", normalized score is " << normalized_score);
  ROS_DEBUG_STREAM("probabilistic score is " << probabilistic_score
                   << ", normalized probabilistic score is " << normalized_probabilistic_score);
  return std::make_tuple(score, normalized_score, probabilistic_score, normalized_probabilistic_score);
}

std::tuple<double, double, double, double> OctomapServerExt::computeScore(const OcTreeT* octree) const {
  double score = 0;
  double normalized_score = 0;
  double probabilistic_score = 0;
  double normalized_probabilistic_score = 0;
  std::size_t occupied_count = 0;
  std::size_t free_count = 0;

  if (m_useOnlySurfaceVoxelsForScore) {
    double max_score = 0.0;
    for (const OcTreeKey &key : m_surfaceVoxelKeys) {
      const point3d point = octree->keyToCoord(key);
      if (!isPointInScoreBoundingBox(point)) {
        continue;
      }
      max_score += m_scorePerSurfaceVoxel;
      const OcTreeNodeT *node = octree->search(key);
      float occupancy;
      float observation_count;
      if (node == nullptr) {
        occupancy = 0.5f;
        observation_count = 0;
      }
      else {
        occupancy = node->getOccupancy();
        observation_count = node->getObservationCount();
      }
      double voxel_score;
      double probabilistic_voxel_score;
      std::tie(voxel_score, probabilistic_voxel_score) = computeVoxelScore(occupancy, observation_count, m_scorePerSurfaceVoxel);
      score += voxel_score;
      probabilistic_score += probabilistic_voxel_score;
      // Is voxel already known with certainty?
      if (node != nullptr) {
        if (node->getOccupancy() >= m_voxelOccupiedThreshold || node->getOccupancy() <= m_voxelFreeThreshold) {
          // Update counts
          if (node->getOccupancy() >= m_voxelOccupiedThreshold) {
            ++occupied_count;
          } else {
            ++free_count;
          }
        }
      }
    }
    // Normalize score by total number of surface voxels and score per surface voxel (i.e. between 0 and 1)
    normalized_score = score / max_score;
    normalized_probabilistic_score = probabilistic_score / max_score;
  }
  else {
    // Alternative computation (considering all voxels)
    double max_score = 0.0;
    for (auto it = octree->begin_leafs(); it != octree->end_leafs(); ++it) {
      if (!isPointInScoreBoundingBox(it.getCoordinate())) {
        continue;
      }
      const bool is_surface_voxel = m_surfaceVoxelKeys.find(it.getKey()) != m_surfaceVoxelKeys.end();
      if (is_surface_voxel) {
        max_score += m_scorePerSurfaceVoxel;
      }

      double voxel_score;
      double probabilistic_voxel_score;
      std::tie(voxel_score, probabilistic_voxel_score) = computeVoxelScore(
              it->getOccupancy(), it->getObservationCount(), is_surface_voxel);
      score += voxel_score;
      probabilistic_score += probabilistic_voxel_score;

      // Is voxel already known with certainty?
      if (it->getOccupancy() >= m_voxelOccupiedThreshold || it->getOccupancy() <= m_voxelFreeThreshold) {
        // Update counts
        if (it->getOccupancy() >= m_voxelOccupiedThreshold) {
          ++occupied_count;
        }
        else {
          ++free_count;
        }
      }
    }
    // Normalized score makes not too much sense in this context. Can be higher than 1.
    normalized_score = score / max_score;
    normalized_probabilistic_score = probabilistic_score / max_score;
  }
  ROS_DEBUG_STREAM("occupied voxels: " << occupied_count << ", free voxels: " << free_count);
  ROS_DEBUG_STREAM("score is " << score << ", normalized score is " << normalized_score);
  ROS_DEBUG_STREAM("probabilistic score is " << probabilistic_score
                                             << ", normalized probabilistic score is " << normalized_probabilistic_score);
  return std::make_tuple(score, normalized_score, probabilistic_score, normalized_probabilistic_score);
}

void OctomapServerExt::filterHeightPointCloud(PCLPointCloud& pc) {
  // set up filter for height range, also removes NANs:
  pcl::PassThrough<PCLPoint> pass_x;
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(m_pointcloudMinX, m_pointcloudMaxX);
  pcl::PassThrough<PCLPoint> pass_y;
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(m_pointcloudMinY, m_pointcloudMaxY);
  pcl::PassThrough<PCLPoint> pass_z;
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(m_pointcloudMinZ, m_pointcloudMaxZ);

  // just filter height range:
  pass_x.setInputCloud(pc.makeShared());
  pass_x.filter(pc);
  pass_y.setInputCloud(pc.makeShared());
  pass_y.filter(pc);
  pass_z.setInputCloud(pc.makeShared());
  pass_z.filter(pc);
}

void OctomapServerExt::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud){
  ros::WallTime startTime = ros::WallTime::now();

  //
  // ground filtering in base frame
  //
  PCLPointCloud pc; // input cloud for filtering and ground-detection
  pcl::fromROSMsg(*cloud, pc);

  tf::StampedTransform sensorToWorldTf;
  try {
    m_tfListener.lookupTransform(m_worldFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToWorldTf);
  } catch(tf::TransformException& ex){
    ROS_ERROR_STREAM( "Transform error of sensor data: " << ex.what() << ", quitting callback");
    return;
  }

  Eigen::Matrix4f sensorToWorld;
  pcl_ros::transformAsMatrix(sensorToWorldTf, sensorToWorld);


  // set up filter for height range, also removes NANs:
  pcl::PassThrough<PCLPoint> pass_x;
  pass_x.setFilterFieldName("x");
  pass_x.setFilterLimits(m_pointcloudMinX, m_pointcloudMaxX);
  pcl::PassThrough<PCLPoint> pass_y;
  pass_y.setFilterFieldName("y");
  pass_y.setFilterLimits(m_pointcloudMinY, m_pointcloudMaxY);
  pcl::PassThrough<PCLPoint> pass_z;
  pass_z.setFilterFieldName("z");
  pass_z.setFilterLimits(m_pointcloudMinZ, m_pointcloudMaxZ);

  PCLPointCloud pc_ground; // segmented ground plane
  PCLPointCloud pc_nonground; // everything else

  if (m_filterGroundPlane){
    tf::StampedTransform sensorToBaseTf, baseToWorldTf;
    try{
      m_tfListener.waitForTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, ros::Duration(0.2));
      m_tfListener.lookupTransform(m_baseFrameId, cloud->header.frame_id, cloud->header.stamp, sensorToBaseTf);
      m_tfListener.lookupTransform(m_worldFrameId, m_baseFrameId, cloud->header.stamp, baseToWorldTf);


    }catch(tf::TransformException& ex){
      ROS_ERROR_STREAM( "Transform error for ground plane filter: " << ex.what() << ", quitting callback.\n"
                        "You need to set the base_frame_id or disable filter_ground.");
    }


    Eigen::Matrix4f sensorToBase, baseToWorld;
    pcl_ros::transformAsMatrix(sensorToBaseTf, sensorToBase);
    pcl_ros::transformAsMatrix(baseToWorldTf, baseToWorld);

    // transform pointcloud from sensor frame to fixed robot frame
    pcl::transformPointCloud(pc, pc, sensorToBase);
    pass_x.setInputCloud(pc.makeShared());
    pass_x.filter(pc);
    pass_y.setInputCloud(pc.makeShared());
    pass_y.filter(pc);
    pass_z.setInputCloud(pc.makeShared());
    pass_z.filter(pc);
    filterGroundPlane(pc, pc_ground, pc_nonground);

    // transform clouds to world frame for insertion
    pcl::transformPointCloud(pc_ground, pc_ground, baseToWorld);
    pcl::transformPointCloud(pc_nonground, pc_nonground, baseToWorld);
  } else {
    // directly transform to map frame:
    pcl::transformPointCloud(pc, pc, sensorToWorld);

    // just filter height range:
    pass_x.setInputCloud(pc.makeShared());
    pass_x.filter(pc);
    pass_y.setInputCloud(pc.makeShared());
    pass_y.filter(pc);
    pass_z.setInputCloud(pc.makeShared());
    pass_z.filter(pc);

    pc_nonground = pc;
    // pc_nonground is empty without ground segmentation
    pc_ground.header = pc.header;
    pc_nonground.header = pc.header;
  }


  insertScan(m_octree, sensorToWorldTf.getOrigin(), pc_ground, pc_nonground, m_updateBBXMin, m_updateBBXMax);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Pointcloud insertion in OctomapServerExt done (%zu+%zu pts (ground/nonground), %f sec)", pc_ground.size(), pc_nonground.size(), total_elapsed);

  publishAll(cloud->header.stamp);
}

OctomapServerExt::PCLPointCloud OctomapServerExt::depthMapToPointCloud(
        const uint32_t height, const uint32_t width, const uint32_t stride,
        const std::vector<float>& depths, const Eigen::Matrix3d& intrinsics,
        const bool downsample_to_grid) const {
  PCLPointCloud pc;
  if (downsample_to_grid) {
    // Need to regenerate spherical grid accelerator?
    if (width != m_spherical_grid_acc.width || height != m_spherical_grid_acc.height
            || intrinsics != m_spherical_grid_acc.intrinsics) {
      ROS_WARN_STREAM("Regenerating spherical grid accelerator structure");
      m_spherical_grid_acc.grid_rays.clear();
      m_spherical_grid_acc.grid_depths.clear();
      m_spherical_grid_acc.used_grid_indices.clear();
      m_spherical_grid_acc.pixel_idx_to_grid_mapping.clear();
      m_spherical_grid_acc.grid_pixel_coordinates.clear();
      m_spherical_grid_acc.width = width;
      m_spherical_grid_acc.height = height;
      m_spherical_grid_acc.intrinsics = intrinsics;
      std::unordered_set<uint32_t> used_grid_indices;;
      std::vector<float> yaws;
      std::vector<float> pitches;
      const float yaw_step = m_spherical_grid_pitch_step;
      const float pitch_step = m_spherical_grid_yaw_step;
      uint32_t num_grid_yaws = 0;
      for (float yaw = - M_PI / 2; yaw <= M_PI / 2; yaw += yaw_step) {
        for (float pitch = - M_PI / 2; pitch <= M_PI / 2; pitch += pitch_step) {
          const float y = std::sin(yaw);
          const float z = std::sin(pitch);
          const float x = std::sqrt(1 - y * y + z * z);
          const float length = std::sqrt(x * x + y * y + z * z);
          const pcl::PointXYZ point(x / length, y / length, z / length);
          m_spherical_grid_acc.grid_rays.push_back(point);
          m_spherical_grid_acc.grid_depths.push_back(std::vector<float>());
          m_spherical_grid_acc.grid_pixel_coordinates.emplace_back(y * intrinsics(0, 0) / x + intrinsics(0, 2),
                                                        z * intrinsics(1, 1) / x + intrinsics(1, 2));
          yaws.push_back(yaw);
          pitches.push_back(pitch);
        }
        ++num_grid_yaws;
      }
      ROS_WARN_STREAM("num_grid_yaws=" << num_grid_yaws);
      const uint32_t num_grid_pitches = m_spherical_grid_acc.grid_rays.size() / num_grid_yaws;
      ROS_WARN_STREAM("num_grid_pitches=" << num_grid_pitches);
      for (uint32_t iy = 0; iy < height; ++iy) {
        for (uint32_t ix = 0; ix < width; ++ix) {
          // Camera coordinate system: x-axis forward, z-axis up, y-axis left
          const float depth = 1.0f;
          const float x = depth;
          const float y = - depth * (ix - intrinsics(0, 2)) / intrinsics(0, 0);
          const float z = - depth * (iy - intrinsics(1, 2)) / intrinsics(1, 1);
          const float yaw = std::atan2(y, x);
          const float pitch = std::atan2(z, x);
          int iyaw = std::round((yaw - M_PI / 2) / yaw_step);
          int ipitch = std::round((pitch - M_PI / 2) / pitch_step);
          if (iyaw < 0) {
            iyaw += num_grid_yaws;
          }
          if (ipitch < 0) {
            ipitch += num_grid_pitches;
          }
          const uint32_t grid_idx = iyaw * num_grid_pitches + ipitch;
          ROS_ASSERT(std::abs(yaws[grid_idx] - yaw) <= yaw_step / 2.0f);
          ROS_ASSERT(std::abs(pitches[grid_idx] - pitch) <= pitch_step / 2.0f);
          m_spherical_grid_acc.pixel_idx_to_grid_mapping.push_back(grid_idx);
          m_spherical_grid_acc.grid_depths[grid_idx].push_back(0);
          used_grid_indices.insert(grid_idx);
          const uint32_t pixel_idx = iy * stride + ix;
          ROS_ASSERT(pixel_idx == m_spherical_grid_acc.pixel_idx_to_grid_mapping.size() - 1);
        }
      }
      ROS_WARN_STREAM("grid_depths.size()=" << m_spherical_grid_acc.grid_depths.size());
      for (uint32_t i = 0; i < m_spherical_grid_acc.grid_depths.size(); ++i) {
        const int num_pixels = m_spherical_grid_acc.grid_depths[i].size();
        m_spherical_grid_acc.grid_depths[i].clear();
        m_spherical_grid_acc.grid_depths[i].reserve(num_pixels);
      }
      for (uint32_t grid_idx : used_grid_indices) {
        m_spherical_grid_acc.used_grid_indices.push_back(grid_idx);
        std::sort(m_spherical_grid_acc.used_grid_indices.begin(), m_spherical_grid_acc.used_grid_indices.end());
      }
    }

    for (uint32_t iy = 0; iy < height; ++iy) {
      for (uint32_t ix = 0; ix < width; ++ix) {
        const uint32_t pixel_idx = iy * stride + ix;
        const float depth = depths[pixel_idx];
        if (depth <= 0) {
          continue;
        }
        const uint32_t grid_idx = m_spherical_grid_acc.pixel_idx_to_grid_mapping[pixel_idx];
        ROS_ASSERT(grid_idx < m_spherical_grid_acc.grid_depths.size());
        m_spherical_grid_acc.grid_depths[grid_idx].push_back(depth);
      }
    }
    for (uint32_t grid_idx : m_spherical_grid_acc.used_grid_indices) {
      const auto it = std::min_element(
              m_spherical_grid_acc.grid_depths[grid_idx].begin(),
              m_spherical_grid_acc.grid_depths[grid_idx].end());
      if (it != m_spherical_grid_acc.grid_depths[grid_idx].end()) {
        const float depth = *it;
        const pcl::PointXYZ &grid_ray = m_spherical_grid_acc.grid_rays[grid_idx];
        pc.push_back(pcl::PointXYZ(depth * grid_ray.x, depth * grid_ray.y, depth * grid_ray.z));
      }
      m_spherical_grid_acc.grid_depths[grid_idx].clear();
    }
  }
  else {
    for (uint32_t iy = 0; iy < height; ++iy) {
      for (uint32_t ix = 0; ix < width; ++ix) {
        const uint32_t idx = iy * stride + ix;
        const float depth = depths[idx];
        if (depth <= 0) {
          continue;
        }
        // Camera coordinate system: x-axis forward, z-axis up, y-axis left
        const float x = depth;
        const float y = - depth * (ix - intrinsics(0, 2)) / intrinsics(0, 0);
        const float z = - depth * (iy - intrinsics(1, 2)) / intrinsics(1, 1);
        pcl::PointXYZ point(x, y, z);
        pc.push_back(point);
      }
    }
    pcl::VoxelGrid<PCLPointCloud::PointType> vg;
    PCLPointCloud::Ptr pc_ptr(new PCLPointCloud());
    *pc_ptr = pc;
    vg.setInputCloud(pc_ptr);
    const double leaf_size = 0.8 * m_octree->getResolution();
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(pc);
    ROS_DEBUG_STREAM("Filtered point cloud from " << pc_ptr->size() << " pts to " << pc.size() << " pts");
  }
  return pc;
}

OctomapServerExt::InsertPointCloudResult OctomapServerExt::insertPointCloud(
        const tf::Point& sensorOriginTf, PCLPointCloud& pc, const bool simulate) {
  ros::WallTime startTime = ros::WallTime::now();

  filterHeightPointCloud(pc);
  ROS_DEBUG("Filter height done (%f sec)", (ros::WallTime::now() - startTime).toSec());
  startTime = ros::WallTime::now();

  const double old_score = m_score;
  const double old_normalized_score = m_normalized_score;
  const double old_probabilistic_score = m_probabilistic_score;
  const double old_normalized_probabilistic_score = m_normalized_probabilistic_score;

  double new_score, new_normalized_score;
  double new_probabilistic_score, new_normalized_probabilistic_score;

  if (simulate) {
    std::tie(new_score, new_normalized_score, new_probabilistic_score, new_normalized_probabilistic_score) =
            computeScoreAfterScanInsertion(m_octree, sensorOriginTf, pc);
  }
  else {
    insertScan(m_octree, sensorOriginTf, pc, m_updateBBXMin, m_updateBBXMax);
    ROS_DEBUG("Insert scan done (%f sec)", (ros::WallTime::now() - startTime).toSec());
    startTime = ros::WallTime::now();

    std::tie(new_score, new_normalized_score, new_probabilistic_score,
             new_normalized_probabilistic_score) = computeScore2(m_octree);
    ROS_DEBUG("Compute score done (%f sec)", (ros::WallTime::now() - startTime).toSec());
    startTime = ros::WallTime::now();
  }

  InsertPointCloudResult res;
  res.score = new_score;
  res.normalized_score = new_normalized_score;
  res.reward = new_score - old_score;
  res.normalized_reward = new_normalized_score - old_normalized_score;
  res.probabilistic_score = new_probabilistic_score;
  res.normalized_probabilistic_score = new_normalized_probabilistic_score;
  res.probabilistic_reward = new_probabilistic_score - old_probabilistic_score;
  res.normalized_probabilistic_reward = new_normalized_probabilistic_score - old_normalized_probabilistic_score;

  startTime = ros::WallTime::now();

  if (!simulate) {
    m_score = new_score;
    m_normalized_score = new_normalized_score;
    m_probabilistic_score = new_probabilistic_score;
    m_normalized_probabilistic_score = new_normalized_probabilistic_score;
  }

  return res;
}

bool OctomapServerExt::insertDepthMapSrv(InsertDepthMap::Request &req, InsertDepthMap::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

//  ROS_DEBUG_STREAM("focal_length_x=" << req.focal_length_x);
//  ROS_DEBUG_STREAM("focal_length_y=" << req.focal_length_y);
//  ROS_DEBUG_STREAM("principal_point_x=" << req.principal_point_x);
//  ROS_DEBUG_STREAM("principal_point_y=" << req.principal_point_y);
//  ROS_DEBUG_STREAM("width=" << req.width);
//  ROS_DEBUG_STREAM("height=" << req.height);
//  ROS_DEBUG_STREAM("stride=" << req.stride);
//  ROS_DEBUG_STREAM("depths.size()=" << req.depths.size());

  Eigen::Matrix3d intrinsics = Eigen::Matrix3d::Zero();
  intrinsics(0, 0) = req.focal_length_x;
  intrinsics(1, 1) = req.focal_length_y;
  intrinsics(0, 2) = req.principal_point_x;
  intrinsics(1, 2) = req.principal_point_y;
  intrinsics(2, 2) = 1;

  PCLPointCloud pc = depthMapToPointCloud(req.height, req.width, req.stride, req.depths,
                                          intrinsics, req.downsample_to_grid);

  if (req.downsample_to_grid) {
    for (const PixelCoordinate& coord : m_spherical_grid_acc.grid_pixel_coordinates) {
      geometry_msgs::Point32 point;
      point.x = coord.x;
      point.y = coord.y;
      point.z = 1;
      res.used_pixel_coordinates.push_back(point);
    }
  }

  ROS_DEBUG("Depth map to pointcloud conversion done (%f sec)", (ros::WallTime::now() - startTime).toSec());

//  ROS_DEBUG_STREAM("pc.size()=" << pc.size());

  tf::Transform sensor_to_world_tf;
  tf::transformMsgToTF(req.sensor_to_world, sensor_to_world_tf);
  Eigen::Matrix4f sensor_to_world;
  pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);

  // directly transform to map frame:
  pcl::transformPointCloud(pc, pc, sensor_to_world);
  // For debugging
//  for (PCLPointCloud::const_iterator it = pc.begin(); it != pc.end(); ++it) {
//    std::cout << "point: (" << it->x << ", " << it->y << ", " << it->z << ")" << std::endl;
//  }

  InsertPointCloudResult ipc_res = insertPointCloud(sensor_to_world_tf.getOrigin(), pc, req.simulate);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Pointcloud insertion in OctomapServerExt done (%zu pts, %f sec)", pc.size(), total_elapsed);

  res.elapsed_seconds = total_elapsed;
  res.score = ipc_res.score;
  res.normalized_score = ipc_res.normalized_score;
  res.reward = ipc_res.reward;
  res.normalized_reward = ipc_res.normalized_reward;
  res.probabilistic_score = ipc_res.probabilistic_score;
  res.normalized_probabilistic_score = ipc_res.normalized_probabilistic_score;
  res.probabilistic_reward = ipc_res.probabilistic_reward;
  res.normalized_probabilistic_reward = ipc_res.normalized_probabilistic_reward;

  publishAll(req.header.stamp);

  return true;
}

bool OctomapServerExt::insertPointCloudSrv(InsertPointCloud::Request &req, InsertPointCloud::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

  PCLPointCloud pc; // input cloud for filtering and ground-detection
  pcl::fromROSMsg(req.point_cloud, pc);

  tf::Transform sensor_to_world_tf;
  tf::transformMsgToTF(req.sensor_to_world, sensor_to_world_tf);
  Eigen::Matrix4f sensor_to_world;
  pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);

  // directly transform to map frame:
  pcl::transformPointCloud(pc, pc, sensor_to_world);

  InsertPointCloudResult ipc_res = insertPointCloud(sensor_to_world_tf.getOrigin(), pc, req.simulate);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Pointcloud insertion in OctomapServerExt done (%zu pts, %f sec)", pc.size(), total_elapsed);

  res.elapsed_seconds = total_elapsed;
  res.score = ipc_res.score;
  res.normalized_score = ipc_res.normalized_score;
  res.reward = ipc_res.reward;
  res.normalized_reward = ipc_res.normalized_reward;
  res.probabilistic_score = ipc_res.probabilistic_score;
  res.normalized_probabilistic_score = ipc_res.normalized_probabilistic_score;
  res.probabilistic_reward = ipc_res.probabilistic_reward;
  res.normalized_probabilistic_reward = ipc_res.normalized_probabilistic_reward;

  publishAll(req.point_cloud.header.stamp);

  return true;
}

void OctomapServerExt::overrideBoundingBox(const octomath::Vector3& min, const octomath::Vector3& max,
                                           const float occupancy, const float observation_count,
                                           const bool densify) {
  const float logodds = octomap::logodds(occupancy);
  const float unknown_logodds_update = logodds - octomap::logodds(0.5);
  const bool lazy_eval = false;
  if (densify) {
    OcTreeKey min_key = m_octree->coordToKey(min);
    OcTreeKey max_key = m_octree->coordToKey(max);
    octomath::Vector3 min_ = m_octree->keyToCoord(min_key);
    octomath::Vector3 max_ = m_octree->keyToCoord(max_key);
    for (key_type a = min_key[0]; a <= max_key[0]; ++a) {
      for (key_type b = min_key[1]; b <= max_key[1]; ++b) {
        for (key_type c = min_key[2]; c <= max_key[2]; ++c) {
          const OcTreeKey key(a, b, c);
          OcTreeNodeT* node = m_octree->search(key);
          if (node == nullptr) {
            m_octree->updateNode(key, unknown_logodds_update, lazy_eval);
            node = m_octree->search(key);
            if (node != nullptr) {
              node->setObservationCount(observation_count);
              node->setMinObservationCount(observation_count);
              node->setMaxObservationCount(observation_count);
            }
            else {
              ROS_ERROR_STREAM("Node not found after creating it");
            }
          }
          else {
            node->setLogOdds(logodds);
            node->setObservationCount(observation_count);
            node->setMinObservationCount(observation_count);
            node->setMaxObservationCount(observation_count);
          }
        }
      }
    }
//    for (double x = min_(0); x <= max_(0); x += m_octree->getResolution()) {
//      for (double y = min_(1); y <= max_(1); y += m_octree->getResolution()) {
//        for (double z = min_(2); z <= max_(2); z += m_octree->getResolution()) {
//          const octomath::Vector3 coord(x, y, z);
//          const OcTreeKey key = m_octree->coordToKey(coord);
//          OcTreeNodeT* node = m_octree->search(key);
//          if (node == nullptr) {
//            m_octree->updateNode(key, unknown_logodds_update, lazy_eval);
//          }
//          else {
//            node->setLogOdds(logodds);
//          }
//        }
//      }
//    }
  }
  else {
    ROS_ERROR_STREAM("Non-dense overriding of bounding boxes not implemented");
//    throw std::runtime_error("Non-dense overriding of bounding boxes not implemented");
//    if (m_octree->getRoot() == nullptr) {
//    }
      // TODO: Need to modify octomap to allow non-dense overriding of bounding boxes.
//      ROS_INFO_STREAM("Tree is empty. Updating center node of bbox");
//      ROS_INFO_STREAM(m_octree->size());
//      ROS_INFO_STREAM(m_octree->size());
//      octomath::Vector3 center = (min + max) * 0.5;
//      ROS_INFO_STREAM("center: " << center);
//      if (!std::isfinite(center(0)) || !std::isfinite(center(1)) || !std::isfinite(center(2))) {
//        center = octomath::Vector3(0, 0, 0);
//      }
//      ROS_INFO_STREAM("center: " << center);
//      m_octree->updateNode(center, unknown_logodds_update, lazy_eval);
//      m_octree->updateInnerOccupancy();
//      if (m_octree->getRoot() == nullptr) {
//        ROS_INFO_STREAM("Tree is still empty.");
//      }
//      for (OcTreeT::leaf_bbx_iterator it = m_octree->begin_leafs_bbx(min, max),
//                   end = m_octree->end_leafs_bbx(); it != end; ++it) {
//        ROS_INFO_STREAM("Bbox> Node: " << reinterpret_cast<uint64_t>(&(*it))
//                                       << ", Depth: " << it.getDepth()
//                                       << ", Key: " << it.getKey()[0] << "," << it.getKey()[1] << "," << it.getKey()[2]
//                                       << ", Position: " << it.getCoordinate()
//                                       << ", Size: " << it.getSize()
//                                       << ", Occupancy: " << it->getOccupancy());
//      }
//      for (OcTreeT::leaf_iterator it = m_octree->begin_leafs(),
//                   end = m_octree->end_leafs(); it != end; ++it) {
//        ROS_INFO_STREAM("Node: " << reinterpret_cast<uint64_t>(&(*it))
//                                       << ", Depth: " << it.getDepth()
//                                       << ", Key: " << it.getKey()[0] << "," << it.getKey()[1] << "," << it.getKey()[2]
//                                       << ", Position: " << it.getCoordinate()
//                                       << ", Size: " << it.getSize()
//                                       << ", Occupancy: " << it->getOccupancy());
//      }
//      const OcTreeNodeT* n1 = m_octree->search(octomath::Vector3(-10, -10, -10));
//      if (n1 == nullptr) {
//        ROS_INFO_STREAM("n1 == nullptr");
//      }
//      else {
//        ROS_INFO_STREAM("n1.occupancy: " << n1->getOccupancy());
//      }
//      const OcTreeNodeT* n2 = m_octree->search(octomath::Vector3(0, 0, 0));
//      if (n2 == nullptr) {
//        ROS_INFO_STREAM("n2 == nullptr");
//      }
//      else {
//        ROS_INFO_STREAM("n2.occupancy: " << n2->getOccupancy());
//      }
//    }
//    else {
//      ROS_INFO_STREAM("Iterating all nodes in bbox");
//      for (OcTreeT::leaf_bbx_iterator it = m_octree->begin_leafs_bbx(min, max),
//                   end = m_octree->end_leafs_bbx(); it != end; ++it) {
//        ROS_INFO_STREAM("Position: " << it.getCoordinate() << ", Size: " << it.getSize() << ", Occupancy: "
//                                     << it->getOccupancy());
//        it->setLogOdds(logodds);
//      }
//    }
  }
  m_octree->updateInnerOccupancy();
}

OctomapServerExt::QueryVoxelsResult OctomapServerExt::queryVoxels(const std::vector<octomath::Vector3>& voxels) {
  QueryVoxelsResult qr;
  qr.num_occupied = 0;
  qr.num_free = 0;
  qr.num_unknown = 0;
  qr.expected_reward = 0;
  // Initialize point cloud
  qr.point_cloud.header.frame_id = m_worldFrameId;
  qr.point_cloud.header.stamp = static_cast<uint32_t>(ros::Time::now().toNSec() / 1000);
  qr.point_cloud.width = voxels.size();
  qr.point_cloud.height = 1;
  qr.point_cloud.is_dense = false;
  qr.point_cloud.points.clear();

  KeySet free_cells, occupied_cells, unknown_cells;
  for (std::size_t i = 0; i < voxels.size(); ++i) {
    const octomath::Vector3& voxel = voxels[i];
    const OcTreeKey key = m_octree->coordToKey(voxel);
    const OcTreeNodeT* node = m_octree->search(key);
    const bool is_known_voxel = node != nullptr;
    if (is_known_voxel) {
      if (node->getOccupancy() >= m_voxelOccupiedThreshold) {
        occupied_cells.insert(key);
      }
      else if (node->getOccupancy() <= m_voxelFreeThreshold) {
        free_cells.insert(key);
      }
      else {
        unknown_cells.insert(key);
      }
    }
    else {
      unknown_cells.insert(key);
    }
    const bool is_surface_voxel = m_surfaceVoxelKeys.find(key) != m_surfaceVoxelKeys.end();
    PointXYZExt point;
    point.x = voxel(0);
    point.y = voxel(1);
    point.z = voxel(2);
    point.is_surface = is_surface_voxel;
    point.is_known = is_known_voxel;
    if (is_known_voxel) {
      point.occupancy = node->getOccupancy();
    }
    else {
      point.occupancy = -1;
    }
    qr.point_cloud.push_back(point);
  }
  qr.num_occupied = occupied_cells.size();
  qr.num_free = free_cells.size();
  qr.num_unknown = unknown_cells.size();
  for (const OcTreeKey& key : unknown_cells) {
    const bool is_surface_voxel = m_surfaceVoxelKeys.find(key) != m_surfaceVoxelKeys.end();
    const bool use_for_reward = !m_useOnlySurfaceVoxelsForScore || is_surface_voxel;
    if (use_for_reward) {
      if (is_surface_voxel) {
        qr.expected_reward += m_scorePerSurfaceVoxel;
      } else {
        qr.expected_reward += m_scorePerVoxel;
      }
    }
  }
  return qr;
}

float OctomapServerExt::interpolateTrilinear(const octomath::Vector3& xyz,
                                             const octomath::Vector3& xyz_000,
                                             const float* grid_size,
                                             const float* values_grid) const {
  const float xd = (xyz(0) - xyz_000(0)) / grid_size[0];
  const float yd = (xyz(1) - xyz_000(1)) / grid_size[1];
  const float zd = (xyz(2) - xyz_000(2)) / grid_size[2];

  const float c00 = values_grid[0] * (1 - zd) + values_grid[1] * zd;
  const float c01 = values_grid[4] * (1 - zd) + values_grid[5] * zd;
  const float c10 = values_grid[2] * (1 - zd) + values_grid[3] * zd;
  const float c11 = values_grid[6] * (1 - zd) + values_grid[7] * zd;

  const float c0 = c00 * (1 - yd) + c10 * yd;
  const float c1 = c01 * (1 - yd) + c11 * yd;

  const float c = c0 * (1 - xd) + c1 * xd;
  const float interp_value = c;
  return interp_value;
}

std::size_t OctomapServerExt::getSubvolumeGridIndex(
        const uint32_t xi, const uint32_t yi, const uint32_t zi,
        const uint32_t size_x, const uint32_t size_y, const uint32_t size_z,
        const uint32_t axis_mode) const {
  std::size_t index;
  if (axis_mode == QuerySubvolumeAxisModeXYZ) {
    index = xi + yi * size_x + zi * size_x * size_y;
  }
  else {
    index = zi + yi * size_z + xi * size_z * size_y;
  }
  return index;
}

OctomapServerExt::QuerySubvolumeResult OctomapServerExt::querySubvolume(
        const octomath::Vector3& center,
        const octomath::Quaternion& orientation,
        const uint32_t level,
        const uint32_t size_x, const uint32_t size_y, const uint32_t size_z,
        const uint32_t axis_mode) {
  QuerySubvolumeResult sr;
  sr.occupancies.clear();
  sr.occupancies.resize(size_x * size_y * size_z);
  sr.observation_certainties.clear();
  sr.observation_certainties.resize(size_x * size_y * size_z);
  sr.min_observation_certainties.clear();
  sr.min_observation_certainties.resize(size_x * size_y * size_z);
  sr.max_observation_certainties.clear();
  sr.max_observation_certainties.resize(size_x * size_y * size_z);

  ROS_INFO_STREAM("center: " << center);

  const float subvolume_voxel_size = static_cast<float>(m_octree->getResolution()) * std::pow<float>(2.0f, level);
//  octomath::Vector3 voxel_min;
  // Note: Integer division is important here!
//  voxel_min(0) = center(0) - subvolume_voxel_size * (size_x / 2);
//  voxel_min(1) = center(1) - subvolume_voxel_size * (size_y / 2);
//  voxel_min(2) = center(2) - subvolume_voxel_size * (size_z / 2);

//  ROS_INFO_STREAM("voxel_min: " << voxel_min);

  ROS_INFO_STREAM("center(2): " << center(2));
  ROS_INFO_STREAM("subvolume_voxel_size: " << subvolume_voxel_size);
  ROS_INFO_STREAM("(size_z / 2): " << (size_z / 2));
  ROS_INFO_STREAM("subvolume_voxel_size * (size_z / 2): " << subvolume_voxel_size * (size_z / 2));

  ROS_INFO_STREAM("Tree depth: " << m_octree->getTreeDepth());

  const float resolution = static_cast<float>(m_octree->getResolution());
  const float grid_size[3] = { resolution, resolution, resolution };
  const float half_voxel_step = 0.5f * m_octree->getResolution();

  // Offsets for trilinear interpolation
  octomath::Vector3 offsets[8];
  offsets[0] = octomath::Vector3(-half_voxel_step, -half_voxel_step, -half_voxel_step);
  offsets[1] = octomath::Vector3(-half_voxel_step, -half_voxel_step, +half_voxel_step);
  offsets[2] = octomath::Vector3(-half_voxel_step, +half_voxel_step, -half_voxel_step);
  offsets[3] = octomath::Vector3(-half_voxel_step, +half_voxel_step, +half_voxel_step);
  offsets[4] = octomath::Vector3(+half_voxel_step, -half_voxel_step, -half_voxel_step);
  offsets[5] = octomath::Vector3(+half_voxel_step, -half_voxel_step, +half_voxel_step);
  offsets[6] = octomath::Vector3(+half_voxel_step, +half_voxel_step, -half_voxel_step);
  offsets[7] = octomath::Vector3(+half_voxel_step, +half_voxel_step, +half_voxel_step);

  // Offset of 000 voxel from center position
  octomath::Vector3 xyz_000_offset(-0.5f * subvolume_voxel_size * (size_x - 1),
                                   -0.5f * subvolume_voxel_size * (size_y - 1),
                                   -0.5f * subvolume_voxel_size * (size_z - 1));

  octomath::Vector3 xyz_local;
  for (uint32_t zi = 0; zi < size_z; ++zi) {
    xyz_local(2) = zi * subvolume_voxel_size + xyz_000_offset(2);
    for (uint32_t yi = 0; yi < size_y; ++yi) {
      xyz_local(1) = yi * subvolume_voxel_size + xyz_000_offset(1);
      for (uint32_t xi = 0; xi < size_x; ++xi) {
        xyz_local(0) = xi * subvolume_voxel_size + xyz_000_offset(0);
        const octomath::Vector3 xyz = orientation.rotate(xyz_local) + center;
        const std::size_t index = getSubvolumeGridIndex(xi, yi, zi, size_x, size_y, size_z, axis_mode);

//        sr.occupancies[index] = 1.0;
//        sr.observation_certainties[index] = 0.0f;
//        sr.min_observation_certainties[index] = 0.0f;
//        sr.max_observation_certainties[index] = 0.0f;
//        if (xi >= size_x / 2) {
//          sr.occupancies[index] = 0.2;
//          if (yi >= size_y / 2) {
//            sr.occupancies[index] = 0.5;
//            if (zi >= size_z / 2) {
//              sr.occupancies[index] = 0.8;
//            }
//          }
//          sr.observation_certainties[index] = 1.0f;
//        }
//        continue;

        if (!isPointInScoreBoundingBox(xyz)) {
          sr.occupancies[index] = m_octree->getClampingThresMin();
//          sr.observation_certainties[index] = 50;
//          sr.min_observation_certainties[index] = 50;
//          sr.max_observation_certainties[index] = 50;
          sr.observation_certainties[index] = 1.0f;
          sr.min_observation_certainties[index] = 1.0f;
          sr.max_observation_certainties[index] = 1.0f;
          continue;
        }

        OcTreeKey keys[8];
        for (unsigned int i = 0; i < 8; ++i) {
          keys[i] = m_octree->coordToKey(xyz + offsets[i]);
        }
        octomath::Vector3 xyz_grid[8];
        for (unsigned int i = 0; i < 8; ++i) {
          xyz_grid[i] = m_octree->keyToCoord(keys[i]);
        }
        const OcTreeNodeT* nodes[8];
        for (unsigned int i = 0; i < 8; ++i) {
          nodes[i] = m_octree->search(keys[i], m_octree->getTreeDepth() - level);
        }
        float occupancies[8];
        float observation_certainties[8];
        float min_observation_certainties[8];
        float max_observation_certainties[8];
        for (unsigned int i = 0; i < 8; ++i) {
          if (nodes[i] == nullptr) {
            occupancies[i] = 0.5f;
            observation_certainties[i] = 0;
            min_observation_certainties[i] = 0;
            max_observation_certainties[i] = 0;
          }
          else {
            occupancies[i] = static_cast<float>(nodes[i]->getOccupancy());
            observation_certainties[i] = computeObservationCertainty(nodes[i]);
            min_observation_certainties[i] = computeObservationCertainty(nodes[i]->getMinObservationCount());
            max_observation_certainties[i] = computeObservationCertainty(nodes[i]->getMaxObservationCount());
          }
        }

        const float occupancy = interpolateTrilinear(xyz, xyz_grid[0], grid_size, occupancies);
        const float observation_certainty = interpolateTrilinear(xyz, xyz_grid[0], grid_size, observation_certainties);
        const float min_observation_certainty = interpolateTrilinear(xyz, xyz_grid[0], grid_size, min_observation_certainties);
        const float max_observation_certainty = interpolateTrilinear(xyz, xyz_grid[0], grid_size, max_observation_certainties);

        sr.occupancies[index] = occupancy;
        sr.observation_certainties[index] = observation_certainty;
        sr.min_observation_certainties[index] = min_observation_certainty;
        sr.max_observation_certainties[index] = max_observation_certainty;
      }
    }
  }

  return sr;
}

OctomapServerExt::RaycastResult OctomapServerExt::raycast(
        const std::vector<Ray>& rays,
        const float occupancy_occupied_threshold,
        const float occupancy_free_threshold,
        const float observation_certainty_occupied_threshold,
        const float observation_certainty_free_threshold,
        const bool ignore_unknown_voxels,
        const float max_raycast_range,
        const bool only_in_score_bounding_box) {
  RaycastResult rr;
  rr.num_hits_occupied = 0;
  rr.num_hits_free = 0;
  rr.num_hits_unknown = 0;
  rr.expected_reward = 0;
  // Initialize point cloud
  rr.point_cloud.header.frame_id = m_worldFrameId;
  rr.point_cloud.header.stamp = static_cast<uint32_t>(ros::Time::now().toNSec() / 1000);
  rr.point_cloud.width = rays.size();
  rr.point_cloud.height = 1;
  rr.point_cloud.is_dense = false;
  rr.point_cloud.points.clear();

  KeySet free_cells, occupied_cells, unknown_cells;
  for (std::size_t i = 0; i < rays.size(); ++i) {
    Ray ray = rays[i];
    float local_max_range = max_raycast_range;
    if (only_in_score_bounding_box) {
      // TODO: This option is quite hacky. It would be much cleaner if we could set a
      // bounding box of the occupancy map to be free and then set the inner score bounding box to unknown.
      ROS_DEBUG_STREAM("Limiting raycast to score bounding box");
      bh::Ray<float> ray_bh(octomapToBh(ray.origin), octomapToBh(ray.direction));
      if (m_scoreBoundingBox.isInside(octomapToBh(ray.origin))) {
        bh::Vector3f intersection;
        const bool intersects = m_scoreBoundingBox.intersects(ray_bh, &intersection);
        local_max_range = (ray_bh.origin - intersection).norm();
        ROS_DEBUG_STREAM("Ray origin in score bounding box. Limiting max range to" << local_max_range);
      }
      else {
        // Move origin to Bbox
        bh::Vector3f intersection;
        const bool intersects = m_scoreBoundingBox.intersects(ray_bh, &intersection);
        if (intersects) {
          ray.origin = bhToOctomap(intersection);
          ROS_DEBUG_STREAM("Moving ray origin to score bounding box: " << ray.origin.x() << ", " << ray.origin.y()
                                                                      << ", " << ray.origin.z());
        }
        else {
          ROS_DEBUG_STREAM("Generating virtual free voxel");
          // Ray is completely outside of score bounding box
          PointXYZExt point;
          octomath::Vector3 hit_point = ray.origin + ray.direction * max_raycast_range;
          if (max_raycast_range < 0) {
            hit_point.x() = bh::signum(ray.direction.x()) * std::numeric_limits<float>::max();
            hit_point.y() = bh::signum(ray.direction.y()) * std::numeric_limits<float>::max();
            hit_point.z() = bh::signum(ray.direction.z()) * std::numeric_limits<float>::max();
          }
          point.x = hit_point.x();
          point.y = hit_point.y();
          point.z = hit_point.z();
          point.is_surface = false;
          point.is_known = true;
          point.occupancy = 0;
          point.observation_certainty = 1;
          rr.point_cloud.push_back(point);
          continue;
        }
      }
    }
    octomath::Vector3 end_point;
    const bool hit_occupied = m_octree->castRay(ray.origin, ray.direction, end_point,
    ignore_unknown_voxels, local_max_range);
    const OcTreeKey end_key = m_octree->coordToKey(end_point);
    const OcTreeNodeT* end_node = m_octree->search(end_key);
    const bool is_known_voxel = end_node != nullptr;
    if (is_known_voxel) {
      const double observation_certainty = computeObservationCertainty(end_node);
      if (end_node->getOccupancy() >= occupancy_occupied_threshold
          && observation_certainty >= observation_certainty_occupied_threshold) {
        occupied_cells.insert(end_key);
      }
      else if (end_node->getOccupancy() <= occupancy_free_threshold
              && observation_certainty >= observation_certainty_free_threshold) {
        free_cells.insert(end_key);
      }
      else {
        unknown_cells.insert(end_key);
      }
    }
    else {
      unknown_cells.insert(end_key);
    }
    const bool is_surface_voxel = m_surfaceVoxelKeys.find(end_key) != m_surfaceVoxelKeys.end();
    PointXYZExt point;
    point.x = end_point(0);
    point.y = end_point(1);
    point.z = end_point(2);
    point.is_surface = is_surface_voxel;
    point.is_known = is_known_voxel;
    if (is_known_voxel) {
      point.occupancy = end_node->getOccupancy();
      point.observation_certainty = computeObservationCertainty(end_node);
    }
    else {
      point.occupancy = -1;
      point.observation_certainty = 0;
    }
    rr.point_cloud.push_back(point);
  }
  rr.num_hits_occupied = occupied_cells.size();
  rr.num_hits_free = free_cells.size();
  rr.num_hits_unknown = unknown_cells.size();
  const double single_observation_certainty = computeObservationCertainty(1);
  for (const OcTreeKey& key : unknown_cells) {
//    const bool is_surface_voxel = m_surfaceVoxelKeys.find(key) != m_surfaceVoxelKeys.end();
//    const bool use_for_reward = !m_useOnlySurfaceVoxelsForScore || is_surface_voxel;
//    if (use_for_reward) {
//      if (is_surface_voxel) {
//        rr.expected_reward += m_scorePerSurfaceVoxel;
//      } else {
//        rr.expected_reward += m_scorePerVoxel;
//      }
//    }
    const OcTreeNodeT* node = m_octree->search(key);
    if (node == nullptr) {
      rr.expected_reward += single_observation_certainty * m_scorePerSurfaceVoxel;
    }
    else {
      const double certainty_diff = computeObservationCertainty(node->getObservationCount())
                                    - computeObservationCertainty(node->getObservationCount() - 1);
      rr.expected_reward += certainty_diff * m_scorePerSurfaceVoxel;
    }
  }
  for (const OcTreeKey& key : occupied_cells) {
    const OcTreeNodeT* node = m_octree->search(key);
    const double certainty_diff = computeObservationCertainty(node->getObservationCount())
                                  - computeObservationCertainty(node->getObservationCount() - 1);
    rr.expected_reward += certainty_diff * m_scorePerSurfaceVoxel;
  }
  return rr;
}

OctomapServerExt::RaycastResult OctomapServerExt::raycastCamera(
        const tf::Transform& sensor_to_world_tf,
        const int height, const int width,
        const float focal_length, const bool ignore_unknown_voxels,
        const float max_range) {
  const point3d sensor_origin  = pointTfToOctomap(sensor_to_world_tf.getOrigin());
  const octomath::Quaternion sensor_orientation = quaternionTfToOctomap(sensor_to_world_tf.getRotation());
  // Camera system: z forward, x right, y down. Transfom sensor_orientation accordingly
//  ROS_INFO_STREAM("sensor_x_axis: " << sensor_orientation.rotate(octomath::Vector3(1, 0, 0)));
//  ROS_INFO_STREAM("sensor_y_axis: " << sensor_orientation.rotate(octomath::Vector3(0, 1, 0)));
//  ROS_INFO_STREAM("sensor_z_axis: " << sensor_orientation.rotate(octomath::Vector3(0, 0, 1)));
  octomath::Quaternion camera_orientation = sensor_orientation;
  camera_orientation = camera_orientation * octomath::Quaternion(octomath::Vector3(0, 0, 1), M_PI / 2);
  camera_orientation = camera_orientation * octomath::Quaternion(octomath::Vector3(1, 0, 0), M_PI / 2);
//  ROS_INFO_STREAM("camera_x_axis: " << camera_orientation.rotate(octomath::Vector3(1, 0, 0)));
//  ROS_INFO_STREAM("camera_y_axis: " << camera_orientation.rotate(octomath::Vector3(0, 1, 0)));
//  ROS_INFO_STREAM("camera_z_axis: " << camera_orientation.rotate(octomath::Vector3(0, 0, 1)));

  if (!m_octree->coordToKeyChecked(sensor_origin, m_updateBBXMin)
      || !m_octree->coordToKeyChecked(sensor_origin, m_updateBBXMax)) {
    ROS_ERROR_STREAM("Could not generate Key for sensor origin " << sensor_origin);
  }

  const float max_raycast_range = max_range;

  RaycastResult rr;
  rr.num_hits_occupied = 0;
  rr.num_hits_free = 0;
  rr.num_hits_unknown = 0;
  rr.expected_reward = 0;
  // Initialize point cloud
  rr.point_cloud.header.frame_id = m_worldFrameId;
  rr.point_cloud.header.stamp = static_cast<uint32_t>(ros::Time::now().toNSec() / 1000);
  rr.point_cloud.width = width;
  rr.point_cloud.height = height;
  rr.point_cloud.is_dense = false;
  rr.point_cloud.points.clear();

  const double center_y = height / 2.0;
  const double center_x = width / 2.0;
  KeySet free_cells, occupied_cells, unknown_cells;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
//      // Camera coordinate system (x-axis forward, z-axis up, y-axis left)
//      const octomath::Vector3 ray_direction_sensor(
//              1,
//              -(x - center_x) / focal_length,
//              (y - center_y) / focal_length);
//      const octomath::Vector3 ray_direction_world = sensor_orientation.rotate(ray_direction_sensor);
      const octomath::Vector3 ray_direction_sensor(
              (x - center_x) / focal_length,
              (y - center_y) / focal_length,
              1.0);
      const octomath::Vector3 ray_direction_world = camera_orientation.rotate(ray_direction_sensor);
      octomath::Vector3 end_point;
      const bool hit_occupied = m_octree->castRay(sensor_origin, ray_direction_world, end_point,
                                                  ignore_unknown_voxels, max_raycast_range);
      const OcTreeKey end_key = m_octree->coordToKey(end_point);
      const OcTreeNodeT* end_node = m_octree->search(end_key);
      const bool is_known_voxel = end_node != nullptr;
      // Update voxel counts
      if (is_known_voxel) {
        if (end_node->getOccupancy() >= m_voxelOccupiedThreshold) {
          occupied_cells.insert(end_key);
        }
        else if (end_node->getOccupancy() <= m_voxelFreeThreshold) {
          free_cells.insert(end_key);
        }
        else {
          unknown_cells.insert(end_key);
        }
      }
      else {
        unknown_cells.insert(end_key);
      }
      const bool is_surface_voxel = m_surfaceVoxelKeys.find(end_key) != m_surfaceVoxelKeys.end();
      PointXYZExt point;
      point.x = end_point(0);
      point.y = end_point(1);
      point.z = end_point(2);
      point.is_surface = is_surface_voxel;
      point.is_known = is_known_voxel;
      if (is_known_voxel) {
        point.occupancy = end_node->getOccupancy();
      }
      else {
        point.occupancy = -1;
      }
      rr.point_cloud.push_back(point);
    }
  }
  rr.num_hits_occupied = occupied_cells.size();
  rr.num_hits_free = free_cells.size();
  rr.num_hits_unknown = unknown_cells.size();
  for (const OcTreeKey& key : unknown_cells) {
    const bool is_surface_voxel = m_surfaceVoxelKeys.find(key) != m_surfaceVoxelKeys.end();
    const bool use_for_reward = !m_useOnlySurfaceVoxelsForScore || is_surface_voxel;
    if (use_for_reward) {
      if (is_surface_voxel) {
        rr.expected_reward += m_scorePerSurfaceVoxel;
      } else {
        rr.expected_reward += m_scorePerVoxel;
      }
    }
  }
  return rr;
}

bool OctomapServerExt::queryVoxelsSrv(QueryVoxels::Request &req, QueryVoxels::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

  std::vector<octomath::Vector3> voxels;
  for (std::size_t i = 0; i < req.voxels.size(); ++i) {
    octomath::Vector3 voxel;
    voxel(0) = req.voxels[i].x;
    voxel(1) = req.voxels[i].y;
    voxel(2) = req.voxels[i].z;
    voxels.push_back(voxel);
  }

  const QueryVoxelsResult qr = queryVoxels(voxels);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Query voxels took %f sec", total_elapsed);

  res.elapsed_seconds = total_elapsed;
  res.num_occupied = qr.num_occupied;
  res.num_free = qr.num_free;
  res.num_unknown = qr.num_unknown;
  res.expected_reward = qr.expected_reward;
  pointCloudExtToROSMsg(qr.point_cloud, res.point_cloud);

  return true;
}

bool OctomapServerExt::querySubvolumeSrv(QuerySubvolume::Request &req, QuerySubvolume::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

  const octomath::Vector3 center(req.center.x, req.center.y, req.center.z);
  const octomath::Quaternion orientation(req.orientation.w, req.orientation.x, req.orientation.y, req.orientation.z);

  const float subvolume_voxel_size = static_cast<float>(m_octree->getResolution()) * std::pow<float>(2.0f, req.level);

  ROS_DEBUG("Subvolume voxel size: %f", subvolume_voxel_size);
  octomath::Vector3 voxel_min;
  // Note: Integer division is important here!
  voxel_min(0) = center(0) - subvolume_voxel_size * (req.size_x / 2);
  voxel_min(1) = center(1) - subvolume_voxel_size * (req.size_y / 2);
  voxel_min(2) = center(2) - subvolume_voxel_size * (req.size_z / 2);
  octomath::Vector3 voxel_max;
  // Note: Integer division is important here!
  voxel_max(0) = center(0) + subvolume_voxel_size * (req.size_x / 2);
  voxel_max(1) = center(1) + subvolume_voxel_size * (req.size_y / 2);
  voxel_max(2) = center(2) + subvolume_voxel_size * (req.size_z / 2);
  const QuerySubvolumeResult qr = querySubvolume(
          center, orientation, req.level, req.size_x, req.size_y, req.size_z, req.axis_mode);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Query subvolume took %f sec", total_elapsed);

  // TODO: Fix min, max voxel
  res.elapsed_seconds = total_elapsed;
  res.voxel_min.x = voxel_min(0);
  res.voxel_min.y = voxel_min(1);
  res.voxel_min.z = voxel_min(2);
  res.voxel_max.x = voxel_max(0);
  res.voxel_max.y = voxel_max(1);
  res.voxel_max.z = voxel_max(2);
  res.voxel_size = subvolume_voxel_size;
  res.occupancies.resize(qr.occupancies.size(), 0.5f);
  std::copy(qr.occupancies.begin(), qr.occupancies.end(), res.occupancies.begin());
  res.observation_certainties.resize(qr.observation_certainties.size(), 0);
  std::copy(qr.observation_certainties.begin(), qr.observation_certainties.end(), res.observation_certainties.begin());
  res.min_observation_certainties.resize(qr.min_observation_certainties.size(), 0);
  std::copy(qr.min_observation_certainties.begin(), qr.min_observation_certainties.end(), res.min_observation_certainties.begin());
  res.max_observation_certainties.resize(qr.max_observation_certainties.size(), 0);
  std::copy(qr.max_observation_certainties.begin(), qr.max_observation_certainties.end(), res.max_observation_certainties.begin());

  // finish pointcloud:
  if (m_subvolumePointCloudPub.getNumSubscribers() > 0) {
    // Offset of 000 voxel from center position
    octomath::Vector3 xyz_000_offset(-0.5f * subvolume_voxel_size * (req.size_x - 1),
                                     -0.5f * subvolume_voxel_size * (req.size_y - 1),
                                     -0.5f * subvolume_voxel_size * (req.size_z - 1));

    float min_occupancy = std::numeric_limits<float>::max();
    float max_occupancy = std::numeric_limits<float>::lowest();
    float min_observation_certainty = std::numeric_limits<float>::max();
    float max_observation_certainty = std::numeric_limits<float>::lowest();
    float min_min_observation_certainty = std::numeric_limits<float>::max();
    float max_min_observation_certainty = std::numeric_limits<float>::lowest();
    float min_max_observation_certainty = std::numeric_limits<float>::max();
    float max_max_observation_certainty = std::numeric_limits<float>::lowest();

    SubvolumePointCloud subvolume_cloud;
    ROS_INFO_STREAM("Publishing subvolume point cloud with voxel size: " << subvolume_voxel_size);
    octomath::Vector3 xyz_local;
    for (uint32_t zi = 0; zi < req.size_z; ++zi) {
      xyz_local(2) = zi * subvolume_voxel_size + xyz_000_offset(2);
      for (uint32_t yi = 0; yi < req.size_y; ++yi) {
        xyz_local(1) = yi * subvolume_voxel_size + xyz_000_offset(1);
        for (uint32_t xi = 0; xi < req.size_x; ++xi) {
          xyz_local(0) = xi * subvolume_voxel_size + xyz_000_offset(0);
          std::size_t index = getSubvolumeGridIndex(xi, yi, zi,
                                                    req.size_x, req.size_y, req.size_z,
                                                    req.axis_mode);
          const octomath::Vector3 xyz = orientation.rotate(xyz_local) + center;
          SubvolumePoint point;
          point.x = xyz(0);
          point.y = xyz(1);
          point.z = xyz(2);
          point.occupancy = qr.occupancies[index];
          point.observation_certainty = qr.observation_certainties[index];
          point.min_observation_certainty = qr.min_observation_certainties[index];
          point.max_observation_certainty = qr.max_observation_certainties[index];
          subvolume_cloud.push_back(point);
          min_occupancy = std::min(point.occupancy, min_occupancy);
          max_occupancy = std::max(point.occupancy, max_occupancy);
          min_observation_certainty = std::min(point.observation_certainty, min_observation_certainty);
          max_observation_certainty = std::max(point.observation_certainty, max_observation_certainty);
          min_min_observation_certainty = std::min(point.min_observation_certainty, min_min_observation_certainty);
          max_min_observation_certainty = std::max(point.min_observation_certainty, max_min_observation_certainty);
          min_max_observation_certainty = std::min(point.max_observation_certainty, min_max_observation_certainty);
          max_max_observation_certainty = std::max(point.max_observation_certainty, max_max_observation_certainty);
        }
      }
    }
    ROS_DEBUG("min_occupancy=%f, max_occupancy=%f", min_occupancy, max_occupancy);
    ROS_DEBUG("min_observation_certainty=%f, max_observation_certainty=%f", min_observation_certainty, max_observation_certainty);
    ROS_DEBUG("min_min_observation_certainty=%f, max_min_observation_certainty=%f", min_min_observation_certainty, max_min_observation_certainty);
    ROS_DEBUG("min_max_observation_certainty=%f, max_max_observation_certainty=%f", min_max_observation_certainty, max_max_observation_certainty);
    sensor_msgs::PointCloud2 ros_cloud;
    subvolumePointCloudToROSMsg(subvolume_cloud, ros_cloud);
    ros_cloud.header.frame_id = m_worldFrameId;
    ros_cloud.header.stamp = ros::Time::now();
    m_subvolumePointCloudPub.publish(ros_cloud);
  }

  return true;
}

bool OctomapServerExt::raycastSrv(Raycast::Request &req, Raycast::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

  std::vector<Ray> rays;
  for (std::size_t i = 0; i < req.rays.size(); ++i) {
    Ray ray;
    ray.origin(0) = req.rays[i].origin.x;
    ray.origin(1) = req.rays[i].origin.y;
    ray.origin(2) = req.rays[i].origin.z;
    ray.direction(0) = req.rays[i].direction.x;
    ray.direction(1) = req.rays[i].direction.y;
    ray.direction(2) = req.rays[i].direction.z;
    rays.push_back(ray);
  }
  const bool ignore_unknown_voxels = req.ignore_unknown_voxels;
  const float max_range = req.max_range;
  const bool only_in_score_bounding_box = req.only_in_score_bounding_box;

  float occupancy_occupied_threshold = m_voxelOccupiedThreshold;
  float occupancy_free_threshold = m_voxelFreeThreshold;
  float observation_certainty_occupied_threshold = 0;
  float observation_certainty_free_threshold = 0;
  if (req.override_thresholds) {
    occupancy_occupied_threshold = req.occupancy_occupied_threshold;
    occupancy_free_threshold = req.occupancy_free_threshold;
    observation_certainty_occupied_threshold = req.observation_certainty_occupied_threshold;
    observation_certainty_free_threshold = req.observation_certainty_free_threshold;
  }

  const RaycastResult rr = raycast(
          rays,
          occupancy_occupied_threshold, occupancy_free_threshold,
          observation_certainty_occupied_threshold, observation_certainty_free_threshold,
          ignore_unknown_voxels,
          max_range,
          only_in_score_bounding_box);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Raycast took %f sec", total_elapsed);

  res.elapsed_seconds = total_elapsed;
  res.num_hits_occupied = rr.num_hits_occupied;
  res.num_hits_free = rr.num_hits_free;
  res.num_hits_unknown = rr.num_hits_unknown;
  res.expected_reward = rr.expected_reward;
  pointCloudExtToROSMsg(rr.point_cloud, res.point_cloud);

  return true;
}

bool OctomapServerExt::raycastCameraSrv(RaycastCamera::Request &req, RaycastCamera::Response &res) {
  ros::WallTime startTime = ros::WallTime::now();

  tf::Transform sensor_to_world_tf;
  tf::transformMsgToTF(req.sensor_to_world, sensor_to_world_tf);
  Eigen::Matrix4f sensor_to_world;
  pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);

  const int height = req.height;
  const int width = req.width;
  const float focal_length = req.focal_length;
  const bool ignore_unknown_voxels = req.ignore_unknown_voxels;
  const float max_range = req.max_range;

  const RaycastResult rr = raycastCamera(
          sensor_to_world_tf, height, width, focal_length, ignore_unknown_voxels, max_range);

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Raycast camera took %f sec", total_elapsed);

  res.elapsed_seconds = total_elapsed;
  res.num_hits_occupied = rr.num_hits_occupied;
  res.num_hits_free = rr.num_hits_free;
  res.num_hits_unknown = rr.num_hits_unknown;
  res.expected_reward = rr.expected_reward;
  pointCloudExtToROSMsg(rr.point_cloud, res.point_cloud);

  return true;
}

void OctomapServerExt::subvolumePointCloudToROSMsg(
        const SubvolumePointCloud& pcl_cloud, sensor_msgs::PointCloud2& ros_cloud) {
  pcl_conversions::fromPCL(pcl_cloud.header.stamp, ros_cloud.header.stamp);
  ros_cloud.header.seq = pcl_cloud.header.seq;
  ros_cloud.header.frame_id = pcl_cloud.header.frame_id;
  ros_cloud.width = pcl_cloud.width;
  ros_cloud.height = pcl_cloud.height;
  ros_cloud.point_step = sizeof(SubvolumePointCloud::PointType);
  ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width;
  ros_cloud.is_dense = pcl_cloud.is_dense;
#if defined(BOOST_BIG_ENDIAN)
  ros_cloud.is_bigendian = true;
#elif defined(BOOST_LITTLE_ENDIAN)
  ros_cloud.is_bigendian = false;
#else
#error "Unable to determine endianness"
#endif
  // Initialize fields array
  ros_cloud.fields.clear ();
//  pcl::for_each_type<typename pcl::traits::fieldList<PointCloudExt::PointType>::type>(
//          pcl::detail::FieldAdder<PointCloudExt::PointType>(ros_cloud.fields));
  SubvolumePointCloud::PointType point;
  sensor_msgs::PointField field;
  field.name = "x";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.x) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "y";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.y) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "z";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.z) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "occupancy";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.occupancy) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "observation_certainty";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.observation_certainty) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "min_observation_certainty";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.min_observation_certainty) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "max_observation_certainty";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.max_observation_certainty) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  // Copy data
  const std::size_t data_size = sizeof(SubvolumePointCloud::PointType) * pcl_cloud.points.size();
  ros_cloud.data.resize(data_size);
  std::memcpy(ros_cloud.data.data(), pcl_cloud.points.data(), data_size);
}

void OctomapServerExt::pointCloudExtToROSMsg(const PointCloudExt& pcl_cloud, sensor_msgs::PointCloud2& ros_cloud) {
  pcl_conversions::fromPCL(pcl_cloud.header.stamp, ros_cloud.header.stamp);
  ros_cloud.header.seq = pcl_cloud.header.seq;
  ros_cloud.header.frame_id = pcl_cloud.header.frame_id;
  ros_cloud.width = pcl_cloud.width;
  ros_cloud.height = pcl_cloud.height;
  ros_cloud.point_step = sizeof(PointCloudExt::PointType);
  ros_cloud.row_step = ros_cloud.point_step * ros_cloud.width;
  ros_cloud.is_dense = pcl_cloud.is_dense;
#if defined(BOOST_BIG_ENDIAN)
  ros_cloud.is_bigendian = true;
#elif defined(BOOST_LITTLE_ENDIAN)
  ros_cloud.is_bigendian = false;
#else
#error "Unable to determine endianness"
#endif
  // Initialize fields array
  ros_cloud.fields.clear ();
//  pcl::for_each_type<typename pcl::traits::fieldList<PointCloudExt::PointType>::type>(
//          pcl::detail::FieldAdder<PointCloudExt::PointType>(ros_cloud.fields));
  PointCloudExt::PointType point;
  sensor_msgs::PointField field;
  field.name = "x";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.x) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "y";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.y) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "z";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.z) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "occupancy";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.occupancy) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "observation_certainty";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.observation_certainty) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::FLOAT32;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "is_surface";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.is_surface) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::UINT8;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  field.name = "is_known";
  field.offset = static_cast<uint32_t>(reinterpret_cast<uint8_t*>(&point.is_known) - reinterpret_cast<uint8_t*>(&point));
  field.datatype = sensor_msgs::PointField::UINT8;
  field.count = 1;
  ros_cloud.fields.push_back(field);
  // Copy data
  const std::size_t data_size = sizeof(PointCloudExt::PointType) * pcl_cloud.points.size();
  ros_cloud.data.resize(data_size);
  std::memcpy(ros_cloud.data.data(), pcl_cloud.points.data(), data_size);
}

std::tuple<double, double> OctomapServerExt::computeVoxelScore(
        const float occupancy, const float observation_count, const double score_per_voxel) const {
  double score;
  double probabilistic_score;
  if (occupancy >= 0.5f) {
    score = (occupancy - 0.5f) / (m_voxelOccupiedThreshold - 0.5f);
    score = score_per_voxel * std::min(score, 1.0);
  }
  else {
    score = 0;
  }
  probabilistic_score = score_per_voxel * computeObservationCertainty(observation_count);

  return std::make_tuple(score, probabilistic_score);
}

std::tuple<double, double> OctomapServerExt::computeVoxelScore(
        const float occupancy, const float observation_count, const bool is_surface_voxel) const {
  double score_per_voxel;
  if (is_surface_voxel) {
    score_per_voxel = m_scorePerSurfaceVoxel;
  }
  else if (m_useOnlySurfaceVoxelsForScore) {
    return std::make_tuple(0.0, 0.0);
  }
  else {
    score_per_voxel = m_scorePerVoxel;
  }
  return computeVoxelScore(occupancy, observation_count, score_per_voxel);
}

std::tuple<double, double> OctomapServerExt::computeUpdateReward(
    const OcTreeT* octree, const octomap::OcTreeKey& key, const bool occupied) const {
  const bool is_surface_voxel = m_surfaceVoxelKeys.count(key) > 0;
  if (!is_surface_voxel && m_useOnlySurfaceVoxelsForScore) {
    return std::make_tuple(0.0, 0.0);
  }

  const OcTreeNodeT* node = octree->search(key, 0);
  float occupancy_log_odds;
  float occupancy;
  float observation_count;
  if (node == nullptr) {
    occupancy = 0.5f;
    occupancy_log_odds = octomap::logodds(0.5f);
    observation_count = 0;
  }
  else {
    occupancy = node->getOccupancy();
    occupancy_log_odds = node->getLogOdds();
    observation_count = node->getObservationCount();
  }

  double voxel_score;
  double probabilistic_score;
  std::tie(voxel_score, probabilistic_score) = computeVoxelScore(occupancy, observation_count, is_surface_voxel);

  double new_occupancy_log_odds = occupancy_log_odds;
  if (occupied) {
    new_occupancy_log_odds += octree->getProbHitLog();
  }
  else {
    new_occupancy_log_odds += octree->getProbMissLog();
  }
  if (new_occupancy_log_odds < octree->getClampingThresMinLog()) {
    new_occupancy_log_odds = octree->getClampingThresMinLog();
  }
  else if (new_occupancy_log_odds > octree->getClampingThresMaxLog()) {
    new_occupancy_log_odds = octree->getClampingThresMaxLog();
  }
  const double new_occupancy = octomap::probability(new_occupancy_log_odds);
  const float new_observation_count = observation_count + 1;

  double new_voxel_score;
  double new_probabilistic_score;
  std::tie(new_voxel_score, new_probabilistic_score)
          = computeVoxelScore(new_occupancy, new_observation_count, is_surface_voxel);

  const double reward = new_voxel_score - voxel_score;
  const double probabilistic_reward = new_probabilistic_score - probabilistic_score;
  return std::make_tuple(reward, probabilistic_reward);
};

std::tuple<double, double> OctomapServerExt::computeRewardAfterScanInsertion(
        const OcTreeT* octree, const tf::Point& sensorOriginTf, const PCLPointCloud& pc,
        const bool ignore_if_maxrange_exceeded) const {
  double reward = 0;
  double probabilistic_reward = 0;

  point3d sensorOrigin = pointTfToOctomap(sensorOriginTf);

  octomap::OcTreeKey key_temp;
  if (!octree->coordToKeyChecked(sensorOrigin, key_temp)
      || !octree->coordToKeyChecked(sensorOrigin, key_temp)) {
    ROS_ERROR_STREAM("Could not generate Key for origin " << sensorOrigin);
  }

  KeySet free_cells, occupied_cells;
  octomap::OcTreeKey bboxMin, bboxMax;
  // insert scan points: free on ray, occupied on endpoint:
  computeScanRayKeys(
          m_octree, sensorOriginTf, pc,
          bboxMin, bboxMax,
          free_cells, occupied_cells);

  // mark free cells only if not already seen occupied in this cloud
  for (KeySet::iterator it = free_cells.begin(), end = free_cells.end(); it != end; ++it) {
    if (occupied_cells.find(*it) != occupied_cells.end()) {
      continue;
    }
    const point3d point = octree->keyToCoord(*it);
    if (!isPointInScoreBoundingBox(point)) {
      continue;
    }
    const bool occupied = false;
    double single_reward;
    double single_probabilistic_reward;
    std::tie(single_reward, single_probabilistic_reward) = computeUpdateReward(octree, *it, occupied);
    reward += single_reward;
    probabilistic_reward += single_probabilistic_reward;
  }

  // now mark all occupied cells:
  for (KeySet::iterator it = occupied_cells.begin(), end = occupied_cells.end(); it != end; it++) {
    const point3d point = octree->keyToCoord(*it);
    if (!isPointInScoreBoundingBox(point)) {
      continue;
    }
    const bool occupied = true;
    double single_reward;
    double single_probabilistic_reward;
    std::tie(single_reward, single_probabilistic_reward) = computeUpdateReward(octree, *it, occupied);
    reward += single_reward;
    probabilistic_reward += single_probabilistic_reward;
  }

  return std::make_tuple(reward, probabilistic_reward);
}

std::tuple<double, double, double, double> OctomapServerExt::computeScoreAfterScanInsertion(
        const OcTreeT* octree, const tf::Point& sensorOriginTf, const PCLPointCloud& pc,
        const bool ignore_if_maxrange_exceeded) const {
  double reward;
  double probabilistic_reward;
  std::tie(reward, probabilistic_reward) = computeRewardAfterScanInsertion(
          octree, sensorOriginTf, pc, ignore_if_maxrange_exceeded);
  const double score = m_score + reward;
  const double probabilistic_score = m_probabilistic_score + probabilistic_reward;

  double max_score = 0.0;
  for (const OcTreeKey &key : m_surfaceVoxelKeys) {
    const point3d point = octree->keyToCoord(key);
    if (!isPointInScoreBoundingBox(point)) {
      continue;
    }
    max_score += m_scorePerSurfaceVoxel;
  }

  const double normalized_score = score / max_score;
  const double normalized_probabilistic_score = probabilistic_score / max_score;

  return std::make_tuple(score, normalized_score, probabilistic_score, normalized_probabilistic_score);
}

void OctomapServerExt::insertScan(OcTreeT* octree, const tf::Point& sensorOriginTf, const PCLPointCloud& pc,
                                  octomap::OcTreeKey& updateBBXMin,
                                  octomap::OcTreeKey& updateBBXMax,
                                  const bool ignore_if_maxrange_exceeded) {
  point3d sensorOrigin = pointTfToOctomap(sensorOriginTf);

  if (!octree->coordToKeyChecked(sensorOrigin, updateBBXMin)
      || !octree->coordToKeyChecked(sensorOrigin, updateBBXMax))
  {
    ROS_ERROR_STREAM("Could not generate Key for origin "<<sensorOrigin);
  }

  KeySet free_cells, occupied_cells;
  computeScanRayKeys(
          m_octree, sensorOriginTf, pc,
          updateBBXMin, updateBBXMax,
          free_cells, occupied_cells);

  // mark free cells only if not seen occupied in this cloud
  for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ++it){
    if (occupied_cells.find(*it) == occupied_cells.end()){
      octree->updateNode(*it, false);
    }
  }

  // now mark all occupied cells:
  for (KeySet::iterator it = occupied_cells.begin(), end=occupied_cells.end(); it!= end; it++) {
    octree->updateNode(*it, true);
  }

  // TODO: eval lazy+updateInner vs. proper insertion
  // non-lazy by default (updateInnerOccupancy() too slow for large maps)
  //octree->updateInnerOccupancy();
  octomap::point3d minPt, maxPt;
  ROS_DEBUG_STREAM("Bounding box keys (before): " << updateBBXMin[0] << " " << updateBBXMin[1] << " " << updateBBXMin[2] << " / " << updateBBXMax[0] << " "<< updateBBXMax[1] << " "<< updateBBXMax[2]);

  // TODO: snap max / min keys to larger voxels by m_maxTreeDepth
//   if (m_maxTreeDepth < 16)
//   {
//      OcTreeKey tmpMin = getIndexKey(updateBBXMin, m_maxTreeDepth); // this should give us the first key at depth m_maxTreeDepth that is smaller or equal to updateBBXMin (i.e. lower left in 2D grid coordinates)
//      OcTreeKey tmpMax = getIndexKey(updateBBXMax, m_maxTreeDepth); // see above, now add something to find upper right
//      tmpMax[0]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[1]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[2]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      updateBBXMin = tmpMin;
//      updateBBXMax = tmpMax;
//   }

  // TODO: we could also limit the bbx to be within the map bounds here (see publishing check)
  minPt = octree->keyToCoord(updateBBXMin);
  maxPt = octree->keyToCoord(updateBBXMax);
  ROS_DEBUG_STREAM("Updated area bounding box: "<< minPt << " - "<<maxPt);
  ROS_DEBUG_STREAM("Bounding box keys (after): " << updateBBXMin[0] << " " << updateBBXMin[1] << " " << updateBBXMin[2] << " / " << updateBBXMax[0] << " "<< updateBBXMax[1] << " "<< updateBBXMax[2]);

  if (m_compressMap)
    octree->prune();

#ifdef COLOR_OCTOMAP_SERVER
  if (colors)
  {
    delete[] colors;
    colors = NULL;
  }
#endif
}

void OctomapServerExt::computeScanRayKeys(
        const OcTreeT* octree, const tf::Point& sensorOriginTf, const PCLPointCloud& pc,
        octomap::OcTreeKey& bboxMin, octomap::OcTreeKey& bboxMax,
        KeySet& free_cells, KeySet& occupied_cells,
        const bool ignore_if_maxrange_exceeded) const {
  point3d sensorOrigin = pointTfToOctomap(sensorOriginTf);

  if (!octree->coordToKeyChecked(sensorOrigin, bboxMin)
      || !octree->coordToKeyChecked(sensorOrigin, bboxMax)) {
    ROS_ERROR_STREAM("Could not generate Key for origin "<<sensorOrigin);
  }

  for (PCLPointCloud::const_iterator it = pc.begin(); it != pc.end(); ++it){
    const point3d point(it->x, it->y, it->z);
    const double distance = (point - sensorOrigin).norm();
    // maxrange check
    if (m_maxRange < 0.0 || distance <= m_maxRange) {
      // free cells
      if (octree->computeRayKeys(sensorOrigin, point, m_keyRay)){
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());
      }
      // occupied endpoint
      OcTreeKey key;
      if (octree->coordToKeyChecked(point, key)) {
        occupied_cells.insert(key);

        updateMinKey(key, bboxMin);
        updateMaxKey(key, bboxMax);
      }
    }
    else if (!ignore_if_maxrange_exceeded) {
      // ray longer than maxrange:;
      const point3d new_end = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
      if (octree->computeRayKeys(sensorOrigin, new_end, m_keyRay)){
        free_cells.insert(m_keyRay.begin(), m_keyRay.end());

        // We should only use the voxel at the truncated end point if we are certain that the ray ended further away.
        if (m_maxRange - distance >= octree->getResolution()) {
          octomap::OcTreeKey endKey;
          if (octree->coordToKeyChecked(new_end, endKey)) {
            free_cells.insert(endKey);
            updateMinKey(endKey, bboxMin);
            updateMaxKey(endKey, bboxMax);
          } else {
            ROS_ERROR_STREAM("Could not generate Key for endpoint " << new_end);
          }
        }
      }
    }
  }
}

void OctomapServerExt::insertScan(OcTreeT* octree, const tf::Point& sensorOriginTf,
                                  const PCLPointCloud& ground, const PCLPointCloud& nonground,
                                  octomap::OcTreeKey& updateBBXMin, octomap::OcTreeKey& updateBBXMax) {
  point3d sensorOrigin = pointTfToOctomap(sensorOriginTf);

  if (!octree->coordToKeyChecked(sensorOrigin, updateBBXMin)
    || !octree->coordToKeyChecked(sensorOrigin, updateBBXMax))
  {
    ROS_ERROR_STREAM("Could not generate Key for origin "<<sensorOrigin);
  }

  // instead of direct scan insertion, compute update to filter ground:
  KeySet free_cells, occupied_cells;
  // insert ground points only as free:
  for (PCLPointCloud::const_iterator it = ground.begin(); it != ground.end(); ++it) {
    point3d point(it->x, it->y, it->z);
    // maxrange check
    if ((m_maxRange > 0.0) && ((point - sensorOrigin).norm() > m_maxRange)) {
      point = sensorOrigin + (point - sensorOrigin).normalized() * m_maxRange;
    }

    // only clear space (ground points)
    if (octree->computeRayKeys(sensorOrigin, point, m_keyRay)) {
      free_cells.insert(m_keyRay.begin(), m_keyRay.end());
    }

    octomap::OcTreeKey endKey;
    if (octree->coordToKeyChecked(point, endKey)) {
      updateMinKey(endKey, updateBBXMin);
      updateMaxKey(endKey, updateBBXMax);
    }
    else {
      ROS_ERROR_STREAM("Could not generate Key for endpoint "<<point);
    }
  }

  computeScanRayKeys(
          m_octree, sensorOriginTf, nonground,
          updateBBXMin, updateBBXMax,
          free_cells, occupied_cells);

  // mark free cells only if not seen occupied in this cloud
  for(KeySet::iterator it = free_cells.begin(), end=free_cells.end(); it!= end; ++it){
    if (occupied_cells.find(*it) == occupied_cells.end()){
      octree->updateNode(*it, false);
    }
  }

  // now mark all occupied cells:
  for (KeySet::iterator it = occupied_cells.begin(), end=occupied_cells.end(); it!= end; it++) {
    octree->updateNode(*it, true);
  }

  // TODO: eval lazy+updateInner vs. proper insertion
  // non-lazy by default (updateInnerOccupancy() too slow for large maps)
  //octree->updateInnerOccupancy();
  octomap::point3d minPt, maxPt;
  ROS_DEBUG_STREAM("Bounding box keys (before): " << updateBBXMin[0] << " " <<updateBBXMin[1] << " " << updateBBXMin[2] << " / " <<updateBBXMax[0] << " "<<updateBBXMax[1] << " "<< updateBBXMax[2]);

  // TODO: snap max / min keys to larger voxels by m_maxTreeDepth
//   if (m_maxTreeDepth < 16)
//   {
//      OcTreeKey tmpMin = getIndexKey(updateBBXMin, m_maxTreeDepth); // this should give us the first key at depth m_maxTreeDepth that is smaller or equal to updateBBXMin (i.e. lower left in 2D grid coordinates)
//      OcTreeKey tmpMax = getIndexKey(updateBBXMax, m_maxTreeDepth); // see above, now add something to find upper right
//      tmpMax[0]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[1]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      tmpMax[2]+= octree->getNodeSize( m_maxTreeDepth ) - 1;
//      updateBBXMin = tmpMin;
//      updateBBXMax = tmpMax;
//   }

  // TODO: we could also limit the bbx to be within the map bounds here (see publishing check)
  minPt = octree->keyToCoord(updateBBXMin);
  maxPt = octree->keyToCoord(updateBBXMax);
  ROS_DEBUG_STREAM("Updated area bounding box: "<< minPt << " - "<<maxPt);
  ROS_DEBUG_STREAM("Bounding box keys (after): " << updateBBXMin[0] << " " <<updateBBXMin[1] << " " << updateBBXMin[2] << " / " <<updateBBXMax[0] << " "<<updateBBXMax[1] << " "<< updateBBXMax[2]);

  if (m_compressMap)
    octree->prune();

#ifdef COLOR_OCTOMAP_SERVER
  if (colors)
  {
    delete[] colors;
    colors = NULL;
  }
#endif
}

void OctomapServerExt::publishAll(const ros::Time& rostime){
  ros::WallTime startTime = ros::WallTime::now();
  size_t octomapSize = m_octree->size();
  // TODO: estimate num occ. voxels for size of arrays (reserve)
  if (octomapSize <= 1){
    ROS_WARN("Nothing to publish, octree is empty");
    return;
  }

  bool publishFreeMarkerArray = m_publishFreeSpace && (m_latchedTopics || m_fmarkerPub.getNumSubscribers() > 0);
  bool publishMarkerArray = (m_latchedTopics || m_markerPub.getNumSubscribers() > 0);
  bool publishPointCloud = (m_latchedTopics || m_pointCloudPub.getNumSubscribers() > 0);
  bool publishBinaryMap = (m_latchedTopics || m_binaryMapPub.getNumSubscribers() > 0);
  bool publishFullMap = (m_latchedTopics || m_fullMapPub.getNumSubscribers() > 0);
  m_publish2DMap = (m_latchedTopics || m_mapPub.getNumSubscribers() > 0);

  // init markers for free space:
  visualization_msgs::MarkerArray freeNodesVis;
  // each array stores all cubes of a different size, one for each depth level:
  freeNodesVis.markers.resize(m_treeDepth+1);

  geometry_msgs::Pose pose;
  pose.orientation = tf::createQuaternionMsgFromYaw(0.0);

  // init markers:
  visualization_msgs::MarkerArray occupiedNodesVis;
  // each array stores all cubes of a different size, one for each depth level:
  occupiedNodesVis.markers.resize(m_treeDepth+1);

  // init pointcloud:
  pcl::PointCloud<PCLPoint> pclCloud;

  // call pre-traversal hook:
  handlePreNodeTraversal(rostime);

  // now, traverse all leafs in the tree:
  for (OcTreeT::iterator it = m_octree->begin(m_maxTreeDepth),
      end = m_octree->end(); it != end; ++it)
  {
    bool inUpdateBBX = isInUpdateBBX(it);

    // call general hook:
    handleNode(it);
    if (inUpdateBBX)
      handleNodeInBBX(it);

    if (m_octree->isNodeOccupied(*it)){
      double z = it.getZ();
      if (z > m_occupancyMinZ && z < m_occupancyMaxZ)
      {
        double size = it.getSize();
        double x = it.getX();
        double y = it.getY();
#ifdef COLOR_OCTOMAP_SERVER
        int r = it->getColor().r;
        int g = it->getColor().g;
        int b = it->getColor().b;
#endif

        // Ignore speckles in the map:
        if (m_filterSpeckles && (it.getDepth() == m_treeDepth +1) && isSpeckleNode(it.getKey())){
          ROS_DEBUG("Ignoring single speckle at (%f,%f,%f)", x, y, z);
          continue;
        } // else: current octree node is no speckle, send it out

        handleOccupiedNode(it);
        if (inUpdateBBX)
          handleOccupiedNodeInBBX(it);


        //create marker:
        if (publishMarkerArray){
          unsigned idx = it.getDepth();
          assert(idx < occupiedNodesVis.markers.size());

          geometry_msgs::Point cubeCenter;
          cubeCenter.x = x;
          cubeCenter.y = y;
          cubeCenter.z = z;

          occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
          if (m_useHeightMap){
            double minX, minY, minZ, maxX, maxY, maxZ;
            m_octree->getMetricMin(minX, minY, minZ);
            m_octree->getMetricMax(maxX, maxY, maxZ);

            double h = (1.0 - std::min(std::max((cubeCenter.z-minZ)/ (maxZ - minZ), 0.0), 1.0)) *m_colorFactor;
            occupiedNodesVis.markers[idx].colors.push_back(heightMapColor(h));
          }

#ifdef COLOR_OCTOMAP_SERVER
          if (m_useColoredMap) {
            std_msgs::ColorRGBA _color; _color.r = (r / 255.); _color.g = (g / 255.); _color.b = (b / 255.); _color.a = 1.0; // TODO/EVALUATE: potentially use occupancy as measure for alpha channel?
            occupiedNodesVis.markers[idx].colors.push_back(_color);
          }
#endif
        }

        // insert into pointcloud:
        if (publishPointCloud) {
#ifdef COLOR_OCTOMAP_SERVER
          PCLPoint _point = PCLPoint();
          _point.x = x; _point.y = y; _point.z = z;
          _point.r = r; _point.g = g; _point.b = b;
          pclCloud.push_back(_point);
#else
          pclCloud.push_back(PCLPoint(x, y, z));
#endif
        }

      }
    } else{ // node not occupied => mark as free in 2D map if unknown so far
      double z = it.getZ();
      if (z > m_occupancyMinZ && z < m_occupancyMaxZ)
      {
        handleFreeNode(it);
        if (inUpdateBBX)
          handleFreeNodeInBBX(it);

        if (m_publishFreeSpace){
          double x = it.getX();
          double y = it.getY();

          //create marker for free space:
          if (publishFreeMarkerArray){
            unsigned idx = it.getDepth();
            assert(idx < freeNodesVis.markers.size());

            geometry_msgs::Point cubeCenter;
            cubeCenter.x = x;
            cubeCenter.y = y;
            cubeCenter.z = z;

            freeNodesVis.markers[idx].points.push_back(cubeCenter);
          }
        }

      }
    }
  }

  // call post-traversal hook:
  handlePostNodeTraversal(rostime);

  // finish MarkerArray:
  if (publishMarkerArray){
    for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){
      double size = m_octree->getNodeSize(i);

      occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
      occupiedNodesVis.markers[i].header.stamp = rostime;
      occupiedNodesVis.markers[i].ns = "map";
      occupiedNodesVis.markers[i].id = i;
      occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
      occupiedNodesVis.markers[i].scale.x = size;
      occupiedNodesVis.markers[i].scale.y = size;
      occupiedNodesVis.markers[i].scale.z = size;
      if (!m_useColoredMap)
        occupiedNodesVis.markers[i].color = m_color;


      if (occupiedNodesVis.markers[i].points.size() > 0)
        occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      else
        occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    m_markerPub.publish(occupiedNodesVis);
  }


  // finish FreeMarkerArray:
  if (publishFreeMarkerArray){
    for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){
      double size = m_octree->getNodeSize(i);

      freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
      freeNodesVis.markers[i].header.stamp = rostime;
      freeNodesVis.markers[i].ns = "map";
      freeNodesVis.markers[i].id = i;
      freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
      freeNodesVis.markers[i].scale.x = size;
      freeNodesVis.markers[i].scale.y = size;
      freeNodesVis.markers[i].scale.z = size;
      freeNodesVis.markers[i].color = m_colorFree;


      if (freeNodesVis.markers[i].points.size() > 0)
        freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      else
        freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    m_fmarkerPub.publish(freeNodesVis);
  }


  // finish pointcloud:
  if (publishPointCloud){
    sensor_msgs::PointCloud2 cloud;
    pcl::toROSMsg (pclCloud, cloud);
    cloud.header.frame_id = m_worldFrameId;
    cloud.header.stamp = rostime;
    m_pointCloudPub.publish(cloud);
  }

  if (publishBinaryMap)
    publishBinaryOctoMap(rostime);

  if (publishFullMap)
    publishFullOctoMap(rostime);


  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_DEBUG("Map publishing in OctomapServerExt took %f sec", total_elapsed);

}


bool OctomapServerExt::octomapBinarySrv(OctomapSrv::Request  &req,
                                    OctomapSrv::Response &res)
{
  ros::WallTime startTime = ros::WallTime::now();
  ROS_INFO("Sending binary map data on service request");
  res.map.header.frame_id = m_worldFrameId;
  res.map.header.stamp = ros::Time::now();
  if (!octomap_msgs::binaryMapToMsg(*m_octree, res.map))
    return false;

  double total_elapsed = (ros::WallTime::now() - startTime).toSec();
  ROS_INFO("Binary octomap sent in %f sec", total_elapsed);
  return true;
}

bool OctomapServerExt::octomapFullSrv(OctomapSrv::Request  &req,
                                    OctomapSrv::Response &res)
{
  ROS_INFO("Sending full map data on service request");
  res.map.header.frame_id = m_worldFrameId;
  res.map.header.stamp = ros::Time::now();


  if (!octomap_msgs::fullMapToMsg(*m_octree, res.map))
    return false;

  return true;
}

bool OctomapServerExt::clearBoundingBoxSrv(ClearBoundingBox::Request& req, ClearBoundingBox::Response& resp){
  point3d min = pointMsgToOctomap(req.min);
  point3d max = pointMsgToOctomap(req.max);
  float occupancy = m_octree->getClampingThresMin();
  float observation_count = std::numeric_limits<float>::max();
  overrideBoundingBox(min, max, occupancy, observation_count, req.densify);

  publishAll(ros::Time::now());

  resp.success = true;

  return true;
}

bool OctomapServerExt::overrideBoundingBoxSrv(OverrideBoundingBox::Request& req, OverrideBoundingBox::Response& resp){
  point3d min = pointMsgToOctomap(req.min);
  point3d max = pointMsgToOctomap(req.max);
  overrideBoundingBox(min, max, req.occupancy, req.observation_count, req.densify);

  publishAll(ros::Time::now());

  resp.success = true;

  return true;
}

bool OctomapServerExt::setScoreBoundingBoxSrv(SetScoreBoundingBox::Request& req, SetScoreBoundingBox::Response& resp) {
  m_scoreBoundingBoxMin = pointMsgToOctomap(req.min);
  m_scoreBoundingBoxMax = pointMsgToOctomap(req.max);
  m_scoreBoundingBox = bh::BoundingBox3Df(octomapToBh(m_scoreBoundingBoxMin), octomapToBh(m_scoreBoundingBoxMax));
  ROS_INFO_STREAM("Set score bounding box to (" << m_scoreBoundingBoxMin << ", " << m_scoreBoundingBoxMax << ")");
  resp.success = true;
  return true;
}

bool OctomapServerExt::infoSrv(Info::Request &req, Info::Response &res) {
  res.tree_type = m_octree->getTreeType();
  res.resolution = m_octree->getResolution();
  res.max_range = m_maxRange;
  res.num_nodes = m_octree->size();
  res.num_leaf_nodes = m_octree->getNumLeafNodes();
  res.use_only_surface_voxels_for_score = m_useOnlySurfaceVoxelsForScore;
  res.surface_voxels_filename = m_surfaceVoxelsFilename;
  res.binary_surface_voxels_filename = m_binarySurfaceVoxelsFilename;

  res.occupancy_clamp_min = m_octree->getClampingThresMin();
  res.occupancy_clamp_max = m_octree->getClampingThresMax();
  res.occupancy_log_odds_clamp_min = m_octree->getClampingThresMinLog();
  res.occupancy_log_odds_clamp_max = m_octree->getClampingThresMaxLog();
  res.occupancy_thres = m_octree->getOccupancyThres();
  res.occupancy_log_odds_thres = m_octree->getOccupancyThresLog();
  res.prob_hit = m_octree->getProbHit();
  res.prob_hit_log_odds = m_octree->getProbHitLog();
  res.prob_miss = m_octree->getProbMiss();
  res.prob_miss_log_odds = m_octree->getProbMissLog();

  std::tie(m_score, m_normalized_score, m_probabilistic_score, m_normalized_probabilistic_score) = computeScore(m_octree);
  res.score = m_score;
  res.normalized_score = m_normalized_score;
  res.probabilistic_score = m_probabilistic_score;
  res.normalized_probabilistic_score = m_normalized_probabilistic_score;

  return true;
}

bool OctomapServerExt::resetSrv(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp) {
  visualization_msgs::MarkerArray occupiedNodesVis;
  occupiedNodesVis.markers.resize(m_treeDepth +1);
  ros::Time rostime = ros::Time::now();
  m_octree->clear();
  // clear 2D map:
  m_gridmap.data.clear();
  m_gridmap.info.height = 0.0;
  m_gridmap.info.width = 0.0;
  m_gridmap.info.resolution = 0.0;
  m_gridmap.info.origin.position.x = 0.0;
  m_gridmap.info.origin.position.y = 0.0;

  m_score = 0.0;
  m_normalized_score = 0.0;
  m_probabilistic_score = 0.0;
  m_normalized_probabilistic_score = 0.0;

  ROS_INFO("Cleared octomap");
  publishAll(rostime);

  publishBinaryOctoMap(rostime);
  for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i){

    occupiedNodesVis.markers[i].header.frame_id = m_worldFrameId;
    occupiedNodesVis.markers[i].header.stamp = rostime;
    occupiedNodesVis.markers[i].ns = "map";
    occupiedNodesVis.markers[i].id = i;
    occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
    occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
  }

  m_markerPub.publish(occupiedNodesVis);

  visualization_msgs::MarkerArray freeNodesVis;
  freeNodesVis.markers.resize(m_treeDepth +1);

  for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i){

    freeNodesVis.markers[i].header.frame_id = m_worldFrameId;
    freeNodesVis.markers[i].header.stamp = rostime;
    freeNodesVis.markers[i].ns = "map";
    freeNodesVis.markers[i].id = i;
    freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
    freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
  }
  m_fmarkerPub.publish(freeNodesVis);

  return true;
}

void OctomapServerExt::publishBinaryOctoMap(const ros::Time& rostime) const{

  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  if (octomap_msgs::binaryMapToMsg(*m_octree, map))
    m_binaryMapPub.publish(map);
  else
    ROS_ERROR("Error serializing OctoMap");
}

void OctomapServerExt::publishFullOctoMap(const ros::Time& rostime) const{

  Octomap map;
  map.header.frame_id = m_worldFrameId;
  map.header.stamp = rostime;

  if (octomap_msgs::fullMapToMsg(*m_octree, map))
    m_fullMapPub.publish(map);
  else
    ROS_ERROR("Error serializing OctoMap");

}


void OctomapServerExt::filterGroundPlane(const PCLPointCloud& pc, PCLPointCloud& ground, PCLPointCloud& nonground) const{
  ground.header = pc.header;
  nonground.header = pc.header;

  if (pc.size() < 50){
    ROS_WARN("Pointcloud in OctomapServerExt too small, skipping ground plane extraction");
    nonground = pc;
  } else {
    // plane detection for ground plane removal:
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Create the segmentation object and set up:
    pcl::SACSegmentation<PCLPoint> seg;
    seg.setOptimizeCoefficients (true);
    // TODO: maybe a filtering based on the surface normals might be more robust / accurate?
    seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(200);
    seg.setDistanceThreshold (m_groundFilterDistance);
    seg.setAxis(Eigen::Vector3f(0,0,1));
    seg.setEpsAngle(m_groundFilterAngle);


    PCLPointCloud cloud_filtered(pc);
    // Create the filtering object
    pcl::ExtractIndices<PCLPoint> extract;
    bool groundPlaneFound = false;

    while(cloud_filtered.size() > 10 && !groundPlaneFound){
      seg.setInputCloud(cloud_filtered.makeShared());
      seg.segment (*inliers, *coefficients);
      if (inliers->indices.size () == 0){
        ROS_INFO("PCL segmentation did not find any plane.");

        break;
      }

      extract.setInputCloud(cloud_filtered.makeShared());
      extract.setIndices(inliers);

      if (std::abs(coefficients->values.at(3)) < m_groundFilterPlaneDistance){
        ROS_DEBUG("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(), cloud_filtered.size(),
                  coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
        extract.setNegative (false);
        extract.filter (ground);

        // remove ground points from full pointcloud:
        // workaround for PCL bug:
        if(inliers->indices.size() != cloud_filtered.size()){
          extract.setNegative(true);
          PCLPointCloud cloud_out;
          extract.filter(cloud_out);
          nonground += cloud_out;
          cloud_filtered = cloud_out;
        }

        groundPlaneFound = true;
      } else{
        ROS_DEBUG("Horizontal plane (not ground) found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(), cloud_filtered.size(),
                  coefficients->values.at(0), coefficients->values.at(1), coefficients->values.at(2), coefficients->values.at(3));
        pcl::PointCloud<PCLPoint> cloud_out;
        extract.setNegative (false);
        extract.filter(cloud_out);
        nonground +=cloud_out;
        // debug
        //            pcl::PCDWriter writer;
        //            writer.write<PCLPoint>("nonground_plane.pcd",cloud_out, false);

        // remove current plane from scan for next iteration:
        // workaround for PCL bug:
        if(inliers->indices.size() != cloud_filtered.size()){
          extract.setNegative(true);
          cloud_out.points.clear();
          extract.filter(cloud_out);
          cloud_filtered = cloud_out;
        } else{
          cloud_filtered.points.clear();
        }
      }

    }
    // TODO: also do this if overall starting pointcloud too small?
    if (!groundPlaneFound){ // no plane found or remaining points too small
      ROS_WARN("No ground plane found in scan");

      // do a rough fitlering on height to prevent spurious obstacles
      pcl::PassThrough<PCLPoint> second_pass;
      second_pass.setFilterFieldName("z");
      second_pass.setFilterLimits(-m_groundFilterPlaneDistance, m_groundFilterPlaneDistance);
      second_pass.setInputCloud(pc.makeShared());
      second_pass.filter(ground);

      second_pass.setFilterLimitsNegative (true);
      second_pass.filter(nonground);
    }

    // debug:
    //        pcl::PCDWriter writer;
    //        if (pc_ground.size() > 0)
    //          writer.write<PCLPoint>("ground.pcd",pc_ground, false);
    //        if (pc_nonground.size() > 0)
    //          writer.write<PCLPoint>("nonground.pcd",pc_nonground, false);

  }


}

void OctomapServerExt::handlePreNodeTraversal(const ros::Time& rostime){
  if (m_publish2DMap){
    // init projected 2D map:
    m_gridmap.header.frame_id = m_worldFrameId;
    m_gridmap.header.stamp = rostime;
    nav_msgs::MapMetaData oldMapInfo = m_gridmap.info;

    // TODO: move most of this stuff into c'tor and init map only once (adjust if size changes)
    double minX, minY, minZ, maxX, maxY, maxZ;
    m_octree->getMetricMin(minX, minY, minZ);
    m_octree->getMetricMax(maxX, maxY, maxZ);

    octomap::point3d minPt(minX, minY, minZ);
    octomap::point3d maxPt(maxX, maxY, maxZ);
    octomap::OcTreeKey minKey = m_octree->coordToKey(minPt, m_maxTreeDepth);
    octomap::OcTreeKey maxKey = m_octree->coordToKey(maxPt, m_maxTreeDepth);

    ROS_DEBUG("MinKey: %d %d %d / MaxKey: %d %d %d", minKey[0], minKey[1], minKey[2], maxKey[0], maxKey[1], maxKey[2]);

    // add padding if requested (= new min/maxPts in x&y):
    double halfPaddedX = 0.5*m_minSizeX;
    double halfPaddedY = 0.5*m_minSizeY;
    minX = std::min(minX, -halfPaddedX);
    maxX = std::max(maxX, halfPaddedX);
    minY = std::min(minY, -halfPaddedY);
    maxY = std::max(maxY, halfPaddedY);
    minPt = octomap::point3d(minX, minY, minZ);
    maxPt = octomap::point3d(maxX, maxY, maxZ);

    OcTreeKey paddedMaxKey;
    if (!m_octree->coordToKeyChecked(minPt, m_maxTreeDepth, m_paddedMinKey)){
      ROS_ERROR("Could not create padded min OcTree key at %f %f %f", minPt.x(), minPt.y(), minPt.z());
      return;
    }
    if (!m_octree->coordToKeyChecked(maxPt, m_maxTreeDepth, paddedMaxKey)){
      ROS_ERROR("Could not create padded max OcTree key at %f %f %f", maxPt.x(), maxPt.y(), maxPt.z());
      return;
    }

    ROS_DEBUG("Padded MinKey: %d %d %d / padded MaxKey: %d %d %d", m_paddedMinKey[0], m_paddedMinKey[1], m_paddedMinKey[2], paddedMaxKey[0], paddedMaxKey[1], paddedMaxKey[2]);
    assert(paddedMaxKey[0] >= maxKey[0] && paddedMaxKey[1] >= maxKey[1]);

    m_multires2DScale = 1 << (m_treeDepth - m_maxTreeDepth);
    m_gridmap.info.width = (paddedMaxKey[0] - m_paddedMinKey[0])/m_multires2DScale +1;
    m_gridmap.info.height = (paddedMaxKey[1] - m_paddedMinKey[1])/m_multires2DScale +1;

    int mapOriginX = minKey[0] - m_paddedMinKey[0];
    int mapOriginY = minKey[1] - m_paddedMinKey[1];
    assert(mapOriginX >= 0 && mapOriginY >= 0);

    // might not exactly be min / max of octree:
    octomap::point3d origin = m_octree->keyToCoord(m_paddedMinKey, m_treeDepth);
    double gridRes = m_octree->getNodeSize(m_maxTreeDepth);
    m_projectCompleteMap = (!m_incrementalUpdate || (std::abs(gridRes-m_gridmap.info.resolution) > 1e-6));
    m_gridmap.info.resolution = gridRes;
    m_gridmap.info.origin.position.x = origin.x() - gridRes*0.5;
    m_gridmap.info.origin.position.y = origin.y() - gridRes*0.5;
    if (m_maxTreeDepth != m_treeDepth){
      m_gridmap.info.origin.position.x -= m_res/2.0;
      m_gridmap.info.origin.position.y -= m_res/2.0;
    }

    // workaround for  multires. projection not working properly for inner nodes:
    // force re-building complete map
    if (m_maxTreeDepth < m_treeDepth)
      m_projectCompleteMap = true;


    if(m_projectCompleteMap){
      ROS_DEBUG("Rebuilding complete 2D map");
      m_gridmap.data.clear();
      // init to unknown:
      m_gridmap.data.resize(m_gridmap.info.width * m_gridmap.info.height, -1);

    } else {

       if (mapChanged(oldMapInfo, m_gridmap.info)){
          ROS_DEBUG("2D grid map size changed to %dx%d", m_gridmap.info.width, m_gridmap.info.height);
          adjustMapData(m_gridmap, oldMapInfo);
       }
       nav_msgs::OccupancyGrid::_data_type::iterator startIt;
       size_t mapUpdateBBXMinX = std::max(0, (int(m_updateBBXMin[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
       size_t mapUpdateBBXMinY = std::max(0, (int(m_updateBBXMin[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));
       size_t mapUpdateBBXMaxX = std::min(int(m_gridmap.info.width-1), (int(m_updateBBXMax[0]) - int(m_paddedMinKey[0]))/int(m_multires2DScale));
       size_t mapUpdateBBXMaxY = std::min(int(m_gridmap.info.height-1), (int(m_updateBBXMax[1]) - int(m_paddedMinKey[1]))/int(m_multires2DScale));

       assert(mapUpdateBBXMaxX > mapUpdateBBXMinX);
       assert(mapUpdateBBXMaxY > mapUpdateBBXMinY);

       size_t numCols = mapUpdateBBXMaxX-mapUpdateBBXMinX +1;

       // test for max idx:
       uint max_idx = m_gridmap.info.width*mapUpdateBBXMaxY + mapUpdateBBXMaxX;
       if (max_idx  >= m_gridmap.data.size())
         ROS_ERROR("BBX index not valid: %d (max index %zu for size %d x %d) update-BBX is: [%zu %zu]-[%zu %zu]", max_idx, m_gridmap.data.size(), m_gridmap.info.width, m_gridmap.info.height, mapUpdateBBXMinX, mapUpdateBBXMinY, mapUpdateBBXMaxX, mapUpdateBBXMaxY);

       // reset proj. 2D map in bounding box:
       for (unsigned int j = mapUpdateBBXMinY; j <= mapUpdateBBXMaxY; ++j){
          std::fill_n(m_gridmap.data.begin() + m_gridmap.info.width*j+mapUpdateBBXMinX,
                      numCols, -1);
       }

    }



  }

}

void OctomapServerExt::handlePostNodeTraversal(const ros::Time& rostime){

  if (m_publish2DMap)
    m_mapPub.publish(m_gridmap);
}

void OctomapServerExt::handleOccupiedNode(const OcTreeT::iterator& it){

  if (m_publish2DMap && m_projectCompleteMap){
    update2DMap(it, true);
  }
}

void OctomapServerExt::handleFreeNode(const OcTreeT::iterator& it){

  if (m_publish2DMap && m_projectCompleteMap){
    update2DMap(it, false);
  }
}

void OctomapServerExt::handleOccupiedNodeInBBX(const OcTreeT::iterator& it){

  if (m_publish2DMap && !m_projectCompleteMap){
    update2DMap(it, true);
  }
}

void OctomapServerExt::handleFreeNodeInBBX(const OcTreeT::iterator& it){

  if (m_publish2DMap && !m_projectCompleteMap){
    update2DMap(it, false);
  }
}

void OctomapServerExt::update2DMap(const OcTreeT::iterator& it, bool occupied){

  // update 2D map (occupied always overrides):

  if (it.getDepth() == m_maxTreeDepth){
    unsigned idx = mapIdx(it.getKey());
    if (occupied)
      m_gridmap.data[mapIdx(it.getKey())] = 100;
    else if (m_gridmap.data[idx] == -1){
      m_gridmap.data[idx] = 0;
    }

  } else{
    int intSize = 1 << (m_maxTreeDepth - it.getDepth());
    octomap::OcTreeKey minKey=it.getIndexKey();
    for(int dx=0; dx < intSize; dx++){
      int i = (minKey[0]+dx - m_paddedMinKey[0])/m_multires2DScale;
      for(int dy=0; dy < intSize; dy++){
        unsigned idx = mapIdx(i, (minKey[1]+dy - m_paddedMinKey[1])/m_multires2DScale);
        if (occupied)
          m_gridmap.data[idx] = 100;
        else if (m_gridmap.data[idx] == -1){
          m_gridmap.data[idx] = 0;
        }
      }
    }
  }


}



bool OctomapServerExt::isSpeckleNode(const OcTreeKey&nKey) const {
  OcTreeKey key;
  bool neighborFound = false;
  for (key[2] = nKey[2] - 1; !neighborFound && key[2] <= nKey[2] + 1; ++key[2]){
    for (key[1] = nKey[1] - 1; !neighborFound && key[1] <= nKey[1] + 1; ++key[1]){
      for (key[0] = nKey[0] - 1; !neighborFound && key[0] <= nKey[0] + 1; ++key[0]){
        if (key != nKey){
          OcTreeNodeT* node = m_octree->search(key);
          if (node && m_octree->isNodeOccupied(node)){
            // we have a neighbor => break!
            neighborFound = true;
          }
        }
      }
    }
  }

  return neighborFound;
}

void OctomapServerExt::reconfigureCallback(octomap_server_ext::OctomapServerExtConfig& config, uint32_t level){
  if (m_maxTreeDepth != unsigned(config.max_depth))
    m_maxTreeDepth = unsigned(config.max_depth);
  else{
    m_pointcloudMinZ            = config.pointcloud_min_z;
    m_pointcloudMaxZ            = config.pointcloud_max_z;
    m_occupancyMinZ             = config.occupancy_min_z;
    m_occupancyMaxZ             = config.occupancy_max_z;
    m_filterSpeckles            = config.filter_speckles;
    m_filterGroundPlane         = config.filter_ground;
    m_compressMap               = config.compress_map;
    m_incrementalUpdate         = config.incremental_2D_projection;

    // Parameters with a namespace require an special treatment at the beginning, as dynamic reconfigure
    // will overwrite them because the server is not able to match parameters' names.
    if (m_initConfig){
		// If parameters do not have the default value, dynamic reconfigure server should be updated.
		if(!is_equal(m_groundFilterDistance, 0.04))
          config.ground_filter_distance = m_groundFilterDistance;
		if(!is_equal(m_groundFilterAngle, 0.15))
          config.ground_filter_angle = m_groundFilterAngle;
	    if(!is_equal( m_groundFilterPlaneDistance, 0.07))
          config.ground_filter_plane_distance = m_groundFilterPlaneDistance;
        if(!is_equal(m_maxRange, -1.0))
          config.sensor_model_max_range = m_maxRange;
        if(!is_equal(m_octree->getProbHit(), 0.7))
          config.sensor_model_hit = m_octree->getProbHit();
	    if(!is_equal(m_octree->getProbMiss(), 0.4))
          config.sensor_model_miss = m_octree->getProbMiss();
		if(!is_equal(m_octree->getClampingThresMin(), 0.12))
          config.sensor_model_min = m_octree->getClampingThresMin();
		if(!is_equal(m_octree->getClampingThresMax(), 0.97))
          config.sensor_model_max = m_octree->getClampingThresMax();
        m_initConfig = false;

	    boost::recursive_mutex::scoped_lock reconf_lock(m_config_mutex);
        m_reconfigureServer.updateConfig(config);
    }
    else{
	  m_groundFilterDistance      = config.ground_filter_distance;
      m_groundFilterAngle         = config.ground_filter_angle;
      m_groundFilterPlaneDistance = config.ground_filter_plane_distance;
      m_maxRange                  = config.sensor_model_max_range;
      m_octree->setClampingThresMin(config.sensor_model_min);
      m_octree->setClampingThresMax(config.sensor_model_max);

     // Checking values that might create unexpected behaviors.
      if (is_equal(config.sensor_model_hit, 1.0))
		config.sensor_model_hit -= 1.0e-6;
      m_octree->setProbHit(config.sensor_model_hit);
	  if (is_equal(config.sensor_model_miss, 0.0))
		config.sensor_model_miss += 1.0e-6;
      m_octree->setProbMiss(config.sensor_model_miss);
	}
  }
  publishAll();
}

void OctomapServerExt::adjustMapData(nav_msgs::OccupancyGrid& map, const nav_msgs::MapMetaData& oldMapInfo) const{
  if (map.info.resolution != oldMapInfo.resolution){
    ROS_ERROR("Resolution of map changed, cannot be adjusted");
    return;
  }

  int i_off = int((oldMapInfo.origin.position.x - map.info.origin.position.x)/map.info.resolution +0.5);
  int j_off = int((oldMapInfo.origin.position.y - map.info.origin.position.y)/map.info.resolution +0.5);

  if (i_off < 0 || j_off < 0
      || oldMapInfo.width  + i_off > map.info.width
      || oldMapInfo.height + j_off > map.info.height)
  {
    ROS_ERROR("New 2D map does not contain old map area, this case is not implemented");
    return;
  }

  nav_msgs::OccupancyGrid::_data_type oldMapData = map.data;

  map.data.clear();
  // init to unknown:
  map.data.resize(map.info.width * map.info.height, -1);

  nav_msgs::OccupancyGrid::_data_type::iterator fromStart, fromEnd, toStart;

  for (int j =0; j < int(oldMapInfo.height); ++j ){
    // copy chunks, row by row:
    fromStart = oldMapData.begin() + j*oldMapInfo.width;
    fromEnd = fromStart + oldMapInfo.width;
    toStart = map.data.begin() + ((j+j_off)*m_gridmap.info.width + i_off);
    copy(fromStart, fromEnd, toStart);

//    for (int i =0; i < int(oldMapInfo.width); ++i){
//      map.data[m_gridmap.info.width*(j+j_off) +i+i_off] = oldMapData[oldMapInfo.width*j +i];
//    }

  }

}


std_msgs::ColorRGBA OctomapServerExt::heightMapColor(double h) {

  std_msgs::ColorRGBA color;
  color.a = 1.0;
  // blend over HSV-values (more colors)

  double s = 1.0;
  double v = 1.0;

  h -= floor(h);
  h *= 6;
  int i;
  double m, n, f;

  i = floor(h);
  f = h - i;
  if (!(i & 1))
    f = 1 - f; // if i is even
  m = v * (1 - s);
  n = v * (1 - s * f);

  switch (i) {
    case 6:
    case 0:
      color.r = v; color.g = n; color.b = m;
      break;
    case 1:
      color.r = n; color.g = v; color.b = m;
      break;
    case 2:
      color.r = m; color.g = v; color.b = n;
      break;
    case 3:
      color.r = m; color.g = n; color.b = v;
      break;
    case 4:
      color.r = n; color.g = m; color.b = v;
      break;
    case 5:
      color.r = v; color.g = m; color.b = n;
      break;
    default:
      color.r = 1; color.g = 0.5; color.b = 0.5;
      break;
  }

  return color;
}
}



