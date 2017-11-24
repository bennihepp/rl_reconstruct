#!/bin/bash

ID=$1

ROSCORE_PORT=$((11911+$ID))

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

sed "s/<<<ID>>>/$ID/g" < RLreconMulti.rviz.template > RLreconMulti_${ID}.rviz
rviz -d RLreconMulti_${ID}.rviz

