#!/bin/bash

ID=$1

if [ -z "${ID}" ]; then
  echo "An id must be specified"
  exit 1
fi

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

CAMERA_NAME=line_camera
#CAMERA_NAME=small_camera

SCENE_ID=7
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/${CAMERA_NAME}_buildings${SCENE_ID}.yml

# New scenes
SCENE_ID=21_steps_500
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/scenes/${CAMERA_NAME}_scene_${SCENE_ID}.yml

SAMPLES_PER_FILE=100
NUM_FILES=1000

DRY_RUN=false
#DRY_RUN=true

if [ -z "$GL_DISPLAY" ]; then
  GL_DISPLAY=:0
fi
DISPLAY=:$((10 + $ID))
WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.width --type int`
HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.height --type int`
FOV=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.fov --type float`

VNC_PORT=$((7000 + $ID))
ROSCORE_PORT=$((11911 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

LAUNCH_FILE=octomap_mapping_ext_custom.launch

# Retrieve engine parameters from YAML config
ENGINE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG engine`


if [ "$ENGINE_NAME" == "unreal" ]; then
  OCTOMAP_WAIT_TIME=10
  COLLECT_DATA_WAIT_TIME=20

  UE4=/home/bhepp/src/UnrealEngine_4.17
  UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

  UNREALCV_PORT=$((9900 + $ID))

  WINDOW_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.width --type int`
  WINDOW_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.height --type int`
  FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.fps --type int --default 20`
  SCENE_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_path`
  SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
  SCENE_FILE=$SCENE_PATH/$SCENE_NAME/$SCENE_NAME.uproject
  MAP_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_path`
  MAP_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_name`
  MAP=$MAP_PATH/$MAP_NAME
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  OCTOMAP_WAIT_TIME=5
  COLLECT_DATA_WAIT_TIME=10

  BH_RENDERER_ZMQ_CMD="python $HOME/rl_reconstruct/python/renderer/mesh_renderer_zmq.py"

  BH_RENDERER_ZMQ_PORT=$((22222 + $ID))

  MESH_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.mesh_path`
  MESH_PATH=$HOME/$MESH_PATH
  WINDOW_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.width`
  WINDOW_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.height`
  MODEL_SCALE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.model_scale --type float --default 1`
  MODEL_YAW=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.model_yaw --type float --default 0`
  MODEL_PITCH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.model_pitch --type float --default 0`
  MODEL_ROLL=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.model_roll --type float --default 0`
  WINDOW_VISIBLE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.visible --type bool --default true`
  FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG bh_renderer_zmq.fps --type int --default 10`
else
  echo "Unknown engine: ${ENGINE_NAME}"
  exit 1
fi

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
POINT_CLOUD_FILTER_FACTOR=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.point_cloud_filter_factor --type float`
SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.surface_voxels_filename`
BINARY_SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.binary_surface_voxels_filename`
BINARY_SURFACE_VOXELS_FILENAME=$HOME/$BINARY_SURFACE_VOXELS_FILENAME
NAIVE_SCAN_INSERTION=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.naive_scan_insertion --type bool`
USE_ONLY_SURFACE_VOXELS_FOR_SCORE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.use_only_surface_voxels_for_score --type bool`

CAMERA_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG environment.camera_name`
ENVIRONMENT_CLASS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG environment.class`
#SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
#OUTPUT_PATH="$HOME/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_${ENVIRONMENT_CLASS}_${SCENE_NAME}_${MAP_NAME}_greedy/data_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/$CAMERA_NAME/in_out_grid_16x16x16_0-1-2-3-4_${ENVIRONMENT_CLASS}_greedy/data_${SCENE_ID}_${ID}"

# New scenes
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_16x16x16_0-1-2-3-4_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_32x32x32_0-1-2_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_8x8x8_0-1-2-3-4-5-6-7-8_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_24x24x24_0-1-2_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_64x64x64_0-1_greedy/data_${SCENE_ID}_${ID}"
#OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_128x128x128_0_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_16x16x1_0-1-2-3_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_16x16x16_0-1-2-3_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_8x8x1_0-1-2-3-4_greedy/data_${SCENE_ID}_${ID}"
OUTPUT_PATH="$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes/in_out_grid_32x32x1_0-1-2_greedy/data_${SCENE_ID}_${ID}"

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
tmux send-keys -t $SESSION:0 "$XVNC_BINARY $DISPLAY -rfbport ${VNC_PORT} -geometry ${WINDOW_WIDTH}x${WINDOW_HEIGHT} -SecurityTypes None" C-m

# Start Unreal
#tmux send-keys -t $SESSION:1 "export DISPLAY=$DISPLAY" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxinfo" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxgears" C-m
if [ "$ENGINE_NAME" == "unreal" ]; then
  # tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WINDOW_WIDTH -ResY=$WINDOW_HEIGHT -VSync -UnrealCVPort $UNREALCV_PORT -UnrealCVWidth $WIDTH -UnrealCVHeight $HEIGHT -UnrealCVExitOnFailure true; sleep 1; done" C-m
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun -d $GL_DISPLAY $UE4EDITOR_CMD \\" C-m \
    "\"$SCENE_FILE\" \\" C-m \
    " \"$MAP\" \\" C-m \
    "-game \\" C-m \
    "-FULLSCREEN \\" C-m \
    "-ResX=$WINDOW_WIDTH \\" C-m \
    "-ResY=$WINDOW_HEIGHT \\" C-m \
    "-VSync \\" C-m \
    "-UnrealCVPort $UNREALCV_PORT \\" C-m \
    "-UnrealCVWidth $WIDTH \\" C-m \
    "-UnrealCVHeight $HEIGHT \\" C-m \
    "-UnrealCVExitOnFailure true; \\" C-m \
    "sleep 1; done" C-m
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun -d $GL_DISPLAY $BH_RENDERER_ZMQ_CMD " \
    "--width $WIDTH " \
    "--height $HEIGHT " \
    "--window-width $WINDOW_WIDTH " \
    "--window-height $WINDOW_HEIGHT " \
    "--mesh-filename $MESH_PATH " \
    "--use-msgpack-for-mesh true " \
    "--model-scale $MODEL_SCALE " \
    "--model-yaw $MODEL_YAW " \
    "--model-pitch $MODEL_PITCH " \
    "--model-roll $MODEL_ROLL " \
    "--horz-fov $FOV " \
    "--address 'tcp://127.0.0.1:$BH_RENDERER_ZMQ_PORT' " \
    "--show-window true " \
    "--poll-timeout 1.0 " \
    "--depth-scale 0.02; " \
    "sleep 1; done" C-m
else
  echo "Unknown engine: $ENGINE_NAME"
  exit 1
fi

tmux send-keys -t $SESSION:2 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:2 "roscore -p $ROSCORE_PORT" C-m
tmux send-keys -t $SESSION:3 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:3 "sleep $OCTOMAP_WAIT_TIME" C-m
tmux send-keys -t $SESSION:3 "roslaunch octomap_server_ext $LAUNCH_FILE " \
  "voxel_size:=$VOXEL_SIZE " \
  "max_range:=$MAX_RANGE " \
  "voxel_free_threshold:=$VOXEL_FREE_THRESHOLD " \
  "voxel_occupied_threshold:=$VOXEL_OCCUPIED_THRESHOLD " \
  "observation_count_saturation:=$OBSERVATION_COUNT_SATURATION " \
  "point_cloud_filter_factor:=$POINT_CLOUD_FILTER_FACTOR " \
  "surface_voxels_filename:=\"$SURFACE_VOXELS_FILENAME\" " \
  "binary_surface_voxels_filename:=\"$BINARY_SURFACE_VOXELS_FILENAME\" " \
  "naive_scan_insertion:=\"$NAIVE_SCAN_INSERTION\" " \
  "use_only_surface_voxels_for_score:=$USE_ONLY_SURFACE_VOXELS_FOR_SCORE " \
  "octomap_server_ext_namespace:=octomap_server_ext_${ID}" C-m

tmux send-keys -t $SESSION:4 "workon rl_reconstruct" C-m
tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:4 "sleep $COLLECT_DATA_WAIT_TIME" C-m
tmux send-keys -t $SESSION:4 "while true; do python $HOME/rl_reconstruct/python/reward_learning/collect_data.py " \
  "--dry-run $DRY_RUN " \
  "--output-path $OUTPUT_PATH " \
  "--obs-levels \"$OBS_LEVELS\" " \
  "--obs-sizes \"$OBS_SIZES\" " \
  "--environment-config $ENVIRONMENT_CONFIG " \
  "--client-id $ID " \
  "--samples-per-file $SAMPLES_PER_FILE " \
  "--num-files $NUM_FILES " \
  "--reset-interval $RESET_INTERVAL " \
  "--reset-score-threshold $RESET_SCORE_THRESHOLD " \
  "--collect-only-selected-action false " \
  "--collect-only-depth-image true " \
  "--collect-output-grid false " \
  "--collect-no-images false " \
  "--measure-timing true; " \
  "sleep 1; done" C-m

tmux attach -t $SESSION:4

