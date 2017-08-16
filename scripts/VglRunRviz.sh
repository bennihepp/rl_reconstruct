#!/bin/bash

ROSCORE_PORT=$((11911+$1))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

vglrun rviz -d `dirname $0`/RLrecon.rviz

