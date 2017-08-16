#!/bin/bash

ROSCORE_PORT=$((11911+$1))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

sleep 5

echo "Starting OctomapExt server on port $ROSCORE_PORT"

#roslaunch octomap_server_ext octomap_mapping_ext_collect_data.launch
roslaunch octomap_server_ext octomap_mapping_ext_rl.launch

