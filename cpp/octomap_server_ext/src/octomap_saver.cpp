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

#include <ros/ros.h>
#include <octomap_msgs/conversions.h>
#include <octomap/octomap.h>
#include <fstream>
#include <bh/mLib/mLibCoreMesh.h>
#include <bh/mLib/mLibUtils.h>

#include <octomap_msgs/GetOctomap.h>
using octomap_msgs::GetOctomap;

#define USAGE "\nUSAGE: octomap_saver [-f] <mapfile.[bt|ot]>\n" \
                "  -f: Query for the full occupancy octree, instead of just the compact binary one\n" \
		"  mapfile.bt: filename of map to be saved (.bt: binary tree, .ot: general octree)\n"

using namespace std;
using namespace octomap;

class MapSaver{
public:
  MapSaver(const std::string& mapname, bool full){
    ros::NodeHandle n;
    std::string servname = "octomap_binary";
    if (full)
      servname = "octomap_full";
    ROS_INFO("Requesting the map from %s...", n.resolveName(servname).c_str());
    GetOctomap::Request req;
    GetOctomap::Response resp;
    while(n.ok() && !ros::service::call(servname, req, resp))
    {
      ROS_WARN("Request to %s failed; trying again...", n.resolveName(servname).c_str());
      usleep(1000000);
    }

    if (n.ok()){ // skip when CTRL-C

      AbstractOcTree* tree = octomap_msgs::msgToMap(resp.map);
      AbstractOccupancyOcTree* octree = NULL;
      if (tree){
        octree = dynamic_cast<AbstractOccupancyOcTree*>(tree);
      } else {
        ROS_ERROR("Error creating octree from received message");
        if (resp.map.id == "ColorOcTree")
          ROS_WARN("You requested a binary map for a ColorOcTree - this is currently not supported. Please add -f to request a full map");
      }

      if (octree){
        ROS_INFO("Map received (%zu nodes, %f m res), saving to %s", octree->size(), octree->getResolution(), mapname.c_str());
        
        std::string suffix = mapname.substr(mapname.length()-3, 3);
        if (suffix== ".bt"){ // write to binary file:
          if (!octree->writeBinary(mapname)){
            ROS_ERROR("Error writing to file %s", mapname.c_str());
          }
        } else if (suffix == ".ot"){ // write to full .ot file:
          if (!octree->write(mapname)){
            ROS_ERROR("Error writing to file %s", mapname.c_str());
          }
        } else{
          ROS_ERROR("Unknown file extension, must be either .bt or .ot");
        }

        // Write occupied voxels as mesh
        OcTree* occ_octree = dynamic_cast<OcTree*>(tree);
        const OcTreeKey key1 = occ_octree->coordToKey(0, 0, 0);
        const OcTreeKey key2 = occ_octree->coordToKey(0.2, 0, 0);
        const OcTreeKey key3 = occ_octree->coordToKey(0, 0.2, 0);
        const OcTreeKey key4 = occ_octree->coordToKey(0, 0, 0.2);
        ROS_WARN_STREAM("key1=" << key1[0] << ", " << key1[1] << ", " << key1[2]);
        ROS_WARN_STREAM("key2=" << key2[0] << ", " << key2[1] << ", " << key2[2]);
        ROS_WARN_STREAM("key3=" << key3[0] << ", " << key3[1] << ", " << key3[2]);
        ROS_WARN_STREAM("key4=" << key4[0] << ", " << key4[1] << ", " << key4[2]);
        const octomath::Vector3 coord1 = occ_octree->keyToCoord(key1);
        const octomath::Vector3 coord2 = occ_octree->keyToCoord(key2);
        const octomath::Vector3 coord3 = occ_octree->keyToCoord(key3);
        const octomath::Vector3 coord4 = occ_octree->keyToCoord(key4);
        ROS_WARN_STREAM("coord1=" << coord1(0) << ", " << coord1(1) << ", " << coord1(2));
        ROS_WARN_STREAM("coord2=" << coord2(0) << ", " << coord2(1) << ", " << coord2(2));
        ROS_WARN_STREAM("coord3=" << coord3(0) << ", " << coord3(1) << ", " << coord3(2));
        ROS_WARN_STREAM("coord4=" << coord4(0) << ", " << coord4(1) << ", " << coord4(2));
        if (occ_octree == nullptr) {
          ROS_WARN("Cannot convert octree to occupancy octree (OcTree)");
        }
        else {
          std::string mesh_output_filename = mapname + ".ply";
          std::vector<ml::TriMeshf> output_meshes;
          for (auto it = occ_octree->begin_leafs(); it != occ_octree->end_leafs(); ++it) {
            if (it->getOccupancy() >= occ_octree->getOccupancyThres()) {
              const octomath::Vector3 &voxel = it.getCoordinate();
              ROS_INFO_STREAM("voxel: " << voxel(0) << ", " << voxel(1) << ", " << voxel(2));
              const float size = it.getSize();
              const ml::BoundingBox3f ml_bbox(
                      ml::vec3f(voxel(0) - size / 2, voxel(1) - size / 2, voxel(2) - size / 2),
                      ml::vec3f(voxel(0) + size / 2, voxel(1) + size / 2, voxel(2) + size / 2));
              const ml::TriMeshf output_mesh = ml::Shapesf::box(ml_bbox);
              output_meshes.push_back(output_mesh);
            }
          }
          const ml::TriMeshf output_mesh = ml::Shapesf::unifyMeshes(output_meshes);
          ml::MeshIOf::saveToPLY(mesh_output_filename, output_mesh.getMeshData());
        }

      } else{
        ROS_ERROR("Error reading OcTree from stream");
      }

      delete octree;

    }
  }
};

int main(int argc, char** argv){
  ros::init(argc, argv, "octomap_saver");
  std::string mapFilename("");
  bool fullmap = false;
  if (argc == 3 && strcmp(argv[1], "-f")==0){
    fullmap = true;
    mapFilename = std::string(argv[2]);
  } else if (argc == 2)
    mapFilename = std::string(argv[1]);
  else{
    ROS_ERROR("%s", USAGE);
    exit(1);
  }

  try{
    MapSaver ms(mapFilename, fullmap);
  }catch(std::runtime_error& e){
    ROS_ERROR("octomap_saver exception: %s", e.what());
    exit(2);
  }

  exit(0);
}


