#!/bin/bash

ROSCORE_PORT=11311

UNREALCV_PORT=9000

#GAMEMODE="-game"
GAMEMODE=""

WORKER_CMD=${@:1}

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

function run_unreal()
{
	local UE4EDITOR_CMD=$HOME/UE4Editor
	local PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"

	local WIDTH=640
	local HEIGHT=480

	echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"
	$UE4EDITOR_CMD "$PROJECT_FILE" $GAMEMODE -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -UnrealCVPort $UNREALCV_PORT >> /dev/null 2>&1 &
	unreal_pid=$!
}

function run_roscore()
{
	echo "Starting ros core listening on port $ROSCORE_PORT"
	roscore -p $ROSCORE_PORT >> /dev/null 2>&1 &
	roscore_pid=$!
}

function run_octomap_server_ext()
{
	echo "Starting OctomapExt server on port $ROSCORE_PORT"
	roslaunch octomap_server_ext octomap_mapping_ext.launch &
	octomap_server_ext_pid=$!
}

run_unreal
run_roscore
sleep 5
run_octomap_server_ext

wait $unreal_pid $roscore_pid $octomap_server_ext_pid

