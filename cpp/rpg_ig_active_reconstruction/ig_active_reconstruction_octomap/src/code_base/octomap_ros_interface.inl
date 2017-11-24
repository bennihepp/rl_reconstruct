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

#define TEMPT template<class TREE_TYPE>
#define CSCOPE RosInterface<TREE_TYPE>

#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>

namespace ig_active_reconstruction
{
  
namespace world_representation
{

namespace octomap
{
  TEMPT
  CSCOPE::RosInterface(Config config)
  : nh_(config.nh)
  , world_frame_name_(config.world_frame_name)
  {
    voxel_map_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("occupied_cells_vis_array", 1);
    free_voxel_map_publisher_ = nh_.advertise<visualization_msgs::MarkerArray>("free_cells_vis_array", 1);
  }

TEMPT
  void CSCOPE::publishVoxelMap()
  {
    publishOccupiedVoxelMap();
    publishFreeVoxelMap();
  }

TEMPT
  void CSCOPE::publishOccupiedVoxelMap()
  {
    if( voxel_map_publisher_.getNumSubscribers()==0 )
      return;
    
    visualization_msgs::MarkerArray occupiedNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    occupiedNodesVis.markers.resize(this->link_.octree->getTreeDepth()+1);
        
    std_msgs::ColorRGBA color;
    color.r = 0;
    color.g = 0;
    color.b = 1;
    color.a = 1;
    
    for( typename TREE_TYPE::iterator it = this->link_.octree->begin(), end = this->link_.octree->end(); it!=end; ++it )
    {
      double size = it.getSize();
      double x = it.getX();
      double y = it.getY();
      double z = it.getZ();
      
      geometry_msgs::Point cubeCenter;
      cubeCenter.x = x;
      cubeCenter.y = y;
      cubeCenter.z = z;
      unsigned idx = it.getDepth();
      
//      if( this->link_.octree->isNodeOccupied(*it) )
      if( it->getOccupancy() > 0.5 )
      {
	double size = it.getSize();
	assert(idx < occupiedNodesVis.markers.size());
	
	occupiedNodesVis.markers[idx].points.push_back(cubeCenter);
        color.b = it->getOccupancy();
        color.a = it->getOccupancy();
        occupiedNodesVis.markers[idx].colors.push_back(color);
      }
    }
    
    for (unsigned i= 0; i < occupiedNodesVis.markers.size(); ++i)
    {
      double size = this->link_.octree->getNodeSize(i);
      
      occupiedNodesVis.markers[i].header.frame_id = world_frame_name_;
      occupiedNodesVis.markers[i].header.stamp = ros::Time::now();
      occupiedNodesVis.markers[i].ns = "map";
      occupiedNodesVis.markers[i].id = i;
      occupiedNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
      occupiedNodesVis.markers[i].scale.x = size;
      occupiedNodesVis.markers[i].scale.y = size;
      occupiedNodesVis.markers[i].scale.z = size;
//      occupiedNodesVis.markers[i].color = color;
      
      if (occupiedNodesVis.markers[i].points.size() > 0)
        occupiedNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      else
        occupiedNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }
    
    voxel_map_publisher_.publish(occupiedNodesVis);
  }

TEMPT
  void CSCOPE::publishFreeVoxelMap()
  {
    if( free_voxel_map_publisher_.getNumSubscribers()==0 )
      return;

    visualization_msgs::MarkerArray freeNodesVis;
    // each array stores all cubes of a different size, one for each depth level:
    freeNodesVis.markers.resize(this->link_.octree->getTreeDepth()+1);

    std_msgs::ColorRGBA color;
    color.r = 0;
    color.g = 0.5;
    color.b = 0;
    color.a = 0.5;

    for( typename TREE_TYPE::iterator it = this->link_.octree->begin(), end = this->link_.octree->end(); it!=end; ++it )
    {
      double size = it.getSize();
      double x = it.getX();
      double y = it.getY();
      double z = it.getZ();

      geometry_msgs::Point cubeCenter;
      cubeCenter.x = x;
      cubeCenter.y = y;
      cubeCenter.z = z;
      unsigned idx = it.getDepth();

      if( it->getOccupancy() < 0.5 )
      {
        double size = it.getSize();
        assert(idx < freeNodesVis.markers.size());

        freeNodesVis.markers[idx].points.push_back(cubeCenter);
        color.g = 0.5 - it->getOccupancy();
        color.a = 0.5 - it->getOccupancy();
        freeNodesVis.markers[idx].colors.push_back(color);
      }
    }

    for (unsigned i= 0; i < freeNodesVis.markers.size(); ++i)
    {
      double size = this->link_.octree->getNodeSize(i);

      freeNodesVis.markers[i].header.frame_id = world_frame_name_;
      freeNodesVis.markers[i].header.stamp = ros::Time::now();
      freeNodesVis.markers[i].ns = "map";
      freeNodesVis.markers[i].id = i;
      freeNodesVis.markers[i].type = visualization_msgs::Marker::CUBE_LIST;
      freeNodesVis.markers[i].scale.x = size;
      freeNodesVis.markers[i].scale.y = size;
      freeNodesVis.markers[i].scale.z = size;
//      freeNodesVis.markers[i].color = color;

      if (freeNodesVis.markers[i].points.size() > 0)
        freeNodesVis.markers[i].action = visualization_msgs::Marker::ADD;
      else
        freeNodesVis.markers[i].action = visualization_msgs::Marker::DELETE;
    }

    free_voxel_map_publisher_.publish(freeNodesVis);
  }

}

}

}

#undef CSCOPE
#undef TEMPT