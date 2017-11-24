/* Copyright (c) 2016, Stefan Isler, islerstefan@bluewin.ch
 * (ETH Zurich / Robotics and Perception Group, University of Zurich, Switzerland)
 *
 * This file is part of ig_active_reconstruction, software for information gain based, active reconstruction.
 *
 * ig_active_reconstruction is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * ig_active_reconstruction is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 * Please refer to the GNU Lesser General Public License for details on the license,
 * on <http://www.gnu.org/licenses/>.
*/

#define TEMPT template<class TREE_TYPE, class POINTCLOUD_TYPE>
#define CSCOPE RosPclInput<TREE_TYPE,POINTCLOUD_TYPE>

#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

namespace ig_active_reconstruction
{
  
namespace world_representation
{

namespace octomap
{
  TEMPT
  CSCOPE::RosPclInput( ros::NodeHandle nh, boost::shared_ptr< PclInput<TREE_TYPE,POINTCLOUD_TYPE> > pcl_input, std::string world_frame )
  : nh_(nh)
  , pcl_input_(pcl_input)
  , world_frame_name_(world_frame)
  , tf_listener_(ros::Duration(180))
  {
    pcl_subscriber_ = nh_.subscribe("pcl_input",10,&CSCOPE::insertCloudCallback,this);
    pcl_input_service_ = nh_.advertiseService("pcl_input", &CSCOPE::insertCloudService,this);
    depth_input_service_ = nh_.advertiseService("depth_input", &CSCOPE::insertDepthMapService,this);
    reset_octomap_service_ = nh_.advertiseService("reset_octomap", &CSCOPE::resetOctomapService,this);
  }
  
  TEMPT
  void CSCOPE::addInputDoneSignalCall( boost::function<void()> signal_call )
  {
    signal_call_stack_.push_back(signal_call);
  }
  
  TEMPT
  void CSCOPE::insertCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
  {
    ROS_INFO("Received new pointcloud. Inserting...");
    POINTCLOUD_TYPE pc;
    pcl::fromROSMsg(*cloud, pc);
    
    insertCloud(pc);
    ROS_INFO("Inserted new pointcloud");
  }
  
  TEMPT
  bool CSCOPE::insertCloudService( ig_active_reconstruction_msgs::PclInput::Request& req, ig_active_reconstruction_msgs::PclInput::Response& res)
  {
    ROS_INFO("Received new pointcloud. Inserting...");
    POINTCLOUD_TYPE pc;
    pcl::fromROSMsg(req.pointcloud, pc);
    
    insertCloud(pc);
    
    ROS_INFO("Inserted new pointcloud");
    res.success = true;
    return true;
  }

  TEMPT
  bool CSCOPE::insertDepthMapService( ig_active_reconstruction_msgs::DepthMapInput::Request& req, ig_active_reconstruction_msgs::DepthMapInput::Response& res)
  {
    ROS_INFO("Received new depth map. Inserting...");

    tf::Transform sensor_to_world_tf;
    tf::transformMsgToTF(req.sensor_to_world, sensor_to_world_tf);
    Eigen::Matrix4f sensor_to_world;
    pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);

    Eigen::Transform<double,3,Eigen::Affine> sensor_to_world_transform;
    sensor_to_world_transform = sensor_to_world.cast<double>();

    insertDepthMap(sensor_to_world_transform,
                   req.depths, req.width, req.height, req.stride,
                   req.focal_length_x, req.focal_length_y,
                   req.principal_point_x, req.principal_point_y);

    ROS_INFO("Inserted new depth map");
    res.success = true;
    return true;
  }

  TEMPT
  bool CSCOPE::resetOctomapService( ig_active_reconstruction_msgs::ResetOctomap::Request& req, ig_active_reconstruction_msgs::ResetOctomap::Response& res)
  {
    ROS_INFO("Resetting octomap.");

    pcl_input_->getOctree()->clear();

    res.success = true;
    return true;
  }

TEMPT
  void CSCOPE::issueInputDoneSignals()
  {
    BOOST_FOREACH( boost::function<void()>& call, signal_call_stack_)
    {
      call();
    }
  }
  
  TEMPT
  void CSCOPE::insertCloud( POINTCLOUD_TYPE& pointcloud )
  {
    tf::StampedTransform sensor_to_world_tf;
    try
    {
      tf_listener_.lookupTransform(world_frame_name_, pointcloud.header.frame_id, ros::Time(0), sensor_to_world_tf);
    }
    catch(tf::TransformException& ex)
    {
      ROS_ERROR_STREAM( "RosPclInput<TREE_TYPE,POINTCLOUD_TYPE>::Transform error of sensor data: " << ex.what() << ", quitting callback.");
      return;
    }
    
    Eigen::Matrix4f sensor_to_world;
    pcl_ros::transformAsMatrix(sensor_to_world_tf, sensor_to_world);
    
    Eigen::Transform<double,3,Eigen::Affine> sensor_to_world_transform;
    sensor_to_world_transform = sensor_to_world.cast<double>();
    
    pcl_input_->push(sensor_to_world_transform,pointcloud);

    issueInputDoneSignals();
  }

  TEMPT
  void CSCOPE::insertDepthMap(const Eigen::Transform<double,3,Eigen::Affine>& sensor_to_world_transform,
                              const std::vector<float>& depths,
                              const std::size_t width,
                              const std::size_t height,
                              const std::size_t stride,
                              const float focal_length_x,
                              const float focal_length_y,
                              const std::size_t principal_point_x,
                              const std::size_t principal_point_y)
  {
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
    intrinsics(0, 0) = focal_length_x;
    intrinsics(1, 1) = focal_length_y;
    intrinsics(0, 2) = principal_point_x;
    intrinsics(1, 2) = principal_point_y;
    intrinsics(2, 2) = 1;

    POINTCLOUD_TYPE pointcloud = depthMapToPointCloud(height, width, stride, depths, intrinsics);

    ROS_INFO("Depth map to pointcloud conversion done (%f sec)", (ros::WallTime::now() - startTime).toSec());
    ROS_DEBUG_STREAM("pc.size()=" << pointcloud.size());

    pcl_input_->push(sensor_to_world_transform, pointcloud);

    double total_elapsed = (ros::WallTime::now() - startTime).toSec();
    ROS_INFO("Pointcloud insertion in OctomapServerExt done (%zu pts, %f sec)", pointcloud.size(), total_elapsed);

    issueInputDoneSignals();
  }

  TEMPT
  POINTCLOUD_TYPE CSCOPE::depthMapToPointCloud(
          const uint32_t height, const uint32_t width, const uint32_t stride,
          const std::vector<float>& depths, const Eigen::Matrix3d& intrinsics) const {
    POINTCLOUD_TYPE pc;
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
        typename POINTCLOUD_TYPE::PointType point(x, y, z);
        pc.push_back(point);
      }
    }
    pcl::VoxelGrid<typename POINTCLOUD_TYPE::PointType> vg;
    typename POINTCLOUD_TYPE::Ptr pc_ptr(new POINTCLOUD_TYPE());
    *pc_ptr = pc;
    vg.setInputCloud(pc_ptr);
    const double leaf_size = 0.8 * pcl_input_->getOctree()->getResolution();
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(pc);
    ROS_DEBUG_STREAM("Filtered point cloud from " << pc_ptr->size() << " pts to " << pc.size() << " pts");
    return pc;
  }

}

}

}

#undef CSCOPE
#undef TEMPT