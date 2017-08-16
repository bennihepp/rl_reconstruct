#!/bin/bash

ROSCORE_PORT=11311

UNREALCV_PORT=9000

WORKER_CMD=${@:1}

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

function run_unreal()
{
	local UE4EDITOR_CMD=$HOME/UE4Editor
	local PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"

	local WIDTH=640
	local HEIGHT=480

	echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"
	$UE4EDITOR_CMD "$PROJECT_FILE" -game -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -UnrealCVPort $UNREALCV_PORT >> /dev/null 2>&1 &
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
	roslaunch octomap_server_ext octomap_mapping_ext.launch >> /dev/null 2>&1 &
	octomap_server_ext_pid=$!
}

function run_worker()
{
	echo "Starting worker. Command: ${WORKER_CMD}"
	$WORKER_CMD
	worker_exit_code=$?
}

run_unreal
run_roscore
sleep 10
run_octomap_server_ext
sleep 10
run_worker

echo "Shutting down ROS and Unreal."
kill -s SIGTERM $unreal_pid
kill -s SIGTERM $roscore_pid
kill -s SIGTERM $octomap_server_ext_pid

sleep 5

kill -s SIGKILL $unreal_pid
kill -s SIGKILL $roscore_pid
kill -s SIGKILL $octomap_server_ext_pid

