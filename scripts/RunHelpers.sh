#!/bin/bash

ID=$1

ROSCORE_PORT=$((11911+$ID))

UNREALCV_PORT=$((9900+$ID))

WORKER_CMD=${@:2}

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

function run_unreal()
{
	local UE4EDITOR_CMD=$HOME/UE4Editor
	local PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"

	local WIDTH=640
	local HEIGHT=480

    local WinX=$((100 + $ID * 100))
    local WinY=$((100 + $ID * 100))

	echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"
	$UE4EDITOR_CMD "$PROJECT_FILE" -name=$ID -game -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -WinX=$WinX -WinY=$WinY -UnrealCVPort $UNREALCV_PORT > unreal_logs/log_$ID.txt 2>&1 &
	unreal_pid=$!
}

function run_roscore()
{
	echo "Starting ros core listening on port $ROSCORE_PORT"
	roscore -p $ROSCORE_PORT > ros_logs/core_$ID.txt 2>&1 &
	roscore_pid=$!
}

function run_octomap_server_ext()
{
	echo "Starting OctomapExt server on port $ROSCORE_PORT"
	roslaunch octomap_server_ext octomap_mapping_ext.launch > ros_logs/octomap_server_ext_$ID.txt 2>&1 &
	octomap_server_ext_pid=$!
}

function run_worker()
{
	echo "Starting worker. Command: ${WORKER_CMD}"
	$WORKER_CMD
	worker_exit_code=$?
}

mkdir ros_logs
mkdir unreal_logs

run_unreal
run_roscore
sleep 10
run_octomap_server_ext

