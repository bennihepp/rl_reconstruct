#!/bin/bash

ROSCORE_PORT=$((11911+$1))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

echo "Starting ros core listening on port $ROSCORE_PORT"

roscore -p $ROSCORE_PORT

