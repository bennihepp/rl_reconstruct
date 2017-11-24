#!/bin/bash

ID=$1
WORKER_IDX=$2

ROSCORE_PORT=$((11911+$1))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

CLIENT_ID=$(($ID + $WORKER_IDX + 1))
sed "s/<<<ID>>>/$CLIENT_ID/g" < RLreconMulti.rviz.template > RLreconMulti_${CLIENT_ID}.rviz
rviz -d RLreconMulti_${CLIENT_ID}.rviz

