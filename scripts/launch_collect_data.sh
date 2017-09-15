#!/bin/bash

ID=$1

if [ -z "${ID}" ]; then
  echo "An id must be specified"
  exit 1
fi

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/line_camera_buildings.yml

RECORDS_PER_FILE=100
NUM_FILES=1000

GL_DISPLAY=:0
DISPLAY=:$((10 + $ID))
WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.width --type int`
HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.height --type int`
FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.fps --type int --default 20`
UNREALCV_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.width --type int`
UNREALCV_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.height --type int`
UNREALCV_FOV=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.fov --type float`

VNC_PORT=$((5910 + $ID))
UNREALCV_PORT=$((9900 + $ID))
ROSCORE_PORT=$((11911 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

UE4=/home/bhepp/src/UnrealEngine_4.17
UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

LAUNCH_FILE=octomap_mapping_ext_custom.launch

# Retrieve Unreal Engine parameters from YAML config
SCENE_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_path`
SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
SCENE_FILE=$SCENE_PATH/$SCENE_NAME/$SCENE_NAME.uproject
MAP_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_path`
MAP_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_name`
MAP=$MAP_PATH/$MAP_NAME

# Retrieve data collection parameters from YAML config
RESET_INTERVAL=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG collect_data.reset_interval --type int`
RESET_SCORE_THRESHOLD=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG collect_data.reset_score_threshold --type float`
OBS_LEVELS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG collect_data.obs_levels --type str`
OBS_SIZES=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG collect_data.obs_sizes`
DOWNSAMPLE_TO_GRID=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG collect_data.downsample_to_grid --type bool`

# Retrieve octomap_server_ext parameters from YAML config
VOXEL_SIZE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.voxel_size --type float`
MAX_RANGE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.max_range --type float`
VOXEL_FREE_THRESHOLD=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.voxel_free_threshold --type float`
VOXEL_OCCUPIED_THRESHOLD=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.voxel_occupied_threshold --type float`
OBSERVATION_COUNT_SATURATION=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.observation_count_saturation --type float`
SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.surface_voxels_filename`
BINARY_SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.binary_surface_voxels_filename`
NAIVE_SCAN_INSERTION=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.naive_scan_insertion --type bool`
USE_ONLY_SURFACE_VOXELS_FOR_SCORE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.use_only_surface_voxels_for_score --type bool`

ENVIRONMENT_CLASS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG environment_class`
SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
OUTPUT_PATH="$HOME/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_${ENVIRONMENT_CLASS}_${SCENE_NAME}_${MAP_NAME}_greedy/data_${ID}"

# Copy environment config to output path
mkdir -p $OUTPUT_PATH/config
cp $ENVIRONMENT_CONFIG $OUTPUT_PATH/config/


# Start processes in tmux session

SESSION=collect-$ID

tmux new-session -s $SESSION -n Xvnc -d
tmux new-window -t $SESSION:1 -n UE4 -c $HOME
tmux new-window -t $SESSION:2 -n RosCore -c $HOME
tmux new-window -t $SESSION:3 -n OctomapServerExt -c $HOME
tmux new-window -t $SESSION:4 -n CollectData -c $HOME

# Start VNC server
tmux send-keys -t $SESSION:0 "$XVNC_BINARY $DISPLAY -rfbport ${VNC_PORT} -geometry ${WIDTH}x${HEIGHT} -SecurityTypes None" C-m

# Start Unreal
#tmux send-keys -t $SESSION:1 "export DISPLAY=$DISPLAY" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxinfo" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxgears" C-m
tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -UnrealCVPort $UNREALCV_PORT -UnrealCVWidth $UNREALCV_WIDTH -UnrealCVHeight $UNREALCV_HEIGHT; sleep 1; done" C-m
#tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m

tmux send-keys -t $SESSION:2 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:2 "roscore -p $ROSCORE_PORT" C-m
tmux send-keys -t $SESSION:3 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:3 "sleep 10" C-m
tmux send-keys -t $SESSION:3 "roslaunch octomap_server_ext $LAUNCH_FILE " \
  "voxel_size:=$VOXEL_SIZE " \
  "max_range:=$MAX_RANGE " \
  "voxel_free_threshold:=$VOXEL_FREE_THRESHOLD " \
  "voxel_occupied_threshold:=$VOXEL_OCCUPIED_THRESHOLD " \
  "observation_count_saturation:=$OBSERVATION_COUNT_SATURATION " \
  "surface_voxels_filename:=\"$SURFACE_VOXELS_FILENAME\" " \
  "binary_surface_voxels_filename:=\"$BINARY_SURFACE_VOXELS_FILENAME\" " \
  "naive_scan_insertion:=\"$NAIVE_SCAN_INSERTION\" " \
  "use_only_surface_voxels_for_score:=$USE_ONLY_SURFACE_VOXELS_FOR_SCORE" C-m

tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:4 "sleep 20" C-m
tmux send-keys -t $SESSION:4 "while true; do python $HOME/rl_reconstruct/python/reward_learning/collect_data.py " \
  "--output-path $OUTPUT_PATH " \
  "--obs-levels \"$OBS_LEVELS\" " \
  "--obs-sizes \"$OBS_SIZES\" " \
  "--environment-config $ENVIRONMENT_CONFIG " \
  "--client-id $ID " \
  "--records-per-file $RECORDS_PER_FILE " \
  "--num-files $NUM_FILES " \
  "--reset-interval $RESET_INTERVAL " \
  "--reset-score-threshold $RESET_SCORE_THRESHOLD " \
  "--downsample-to-grid $DOWNSAMPLE_TO_GRID; " \
  "sleep 1; done" C-m

tmux attach -t $SESSION:4

