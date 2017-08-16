#!/bin/bash

ROSCORE_PORT=$((11911+$1))

#SURFACE_VOXEL_FILENAME="/home/t-behepp/Documents/UnrealProjects/CustomScenes/Export/medieval_building/surface_voxels.txt"
SURFACE_VOXEL_FILENAME="/home/t-behepp/Documents/UnrealProjects/CustomScenes/Export/buildings/surface_voxels.txt"
LAUNCH_FILE=octomap_mapping_ext_rl.launch

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

sleep 5

echo "Starting OctomapExt server on port $ROSCORE_PORT"

roslaunch octomap_server_ext $LAUNCH_FILE surface_voxel_filename:=$SURFACE_VOXEL_FILENAME

