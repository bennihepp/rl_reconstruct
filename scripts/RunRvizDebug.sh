#!/bin/bash

ROSCORE_PORT=$((11311))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

rviz -d RLrecon.rviz

