#!/bin/bash

ID=$1

if [ -z "${ID}" ]; then
  echo "A job-id must be specified"
  exit 1
fi

SCENE_ID=$2
if [ -z "${SCENE_ID}" ]; then
  echo "A scene-id must be specified"
  exit 1
fi

CAMERA_NAME=small_camera
PRNG_SEED=41

INPUT_FOLDER=outputs/output_${CAMERA_NAME}_new
INPUT_FOLDER=outputs/output_${CAMERA_NAME}

# New scenes
#SCENE_ID=20
#SCENE_ID=30
INPUT_FOLDER=outputs_scenes/output_${CAMERA_NAME}_fixed
INPUT_FOLDER=outputs_scenes/output_${CAMERA_NAME}_video

RUN_VISUALIZATION=false
#RUN_VISUALIZATION=true

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

# SCENE_ID="1"
# SCENE_ID="2"
# SCENE_ID="3"
# SCENE_ID="3.1"
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/${CAMERA_NAME}_buildings${SCENE_ID}.yml
# ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/small_camera_buildings${SCENE_ID}_EnvironmentNoPitch.yml

# New scenes
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/scenes/${CAMERA_NAME}_scene_${SCENE_ID}.yml

GL_DISPLAY=:0
VNC_DISPLAY=:$((10 + $ID))
WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.width --type int`
HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.height --type int`
FOV=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.fov --type float`

VNC_PORT=$((7000 + $ID))
ROSCORE_PORT=$((11911 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

RVIZ_CONFIG="`dirname $0`/RLrecon.rviz"

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"

LAUNCH_FILE=octomap_mapping_ext_custom.launch

# Retrieve engine parameters from YAML config
ENGINE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG engine`

if [ "$ENGINE_NAME" == "unreal" ]; then
  OCTOMAP_WAIT_TIME=10
  EVALUATION_WAIT_TIME=20

  UE4=/home/bhepp/src/UnrealEngine_4.17
  UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

  UNREALCV_PORT=$((9900 + $ID))

  WINDOW_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.width --type int`
  WINDOW_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.height --type int`
  FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.fps --type int --default 20`
# Retrieve Unreal Engine parameters from YAML config
  SCENE_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_path`
  SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
  SCENE_FILE=$SCENE_PATH/$SCENE_NAME/$SCENE_NAME.uproject
  MAP_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_path`
  MAP_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_name`
  MAP=$MAP_PATH/$MAP_NAME
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  OCTOMAP_WAIT_TIME=5
  EVALUATION_WAIT_TIME=10

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

NUM_EPISODES=50
MAX_FILE_NUM=100

MEASURE_TIMING=true

VISUALIZE=false
#VISUALIZE=true

INTERACTIVE=false
# INTERACTIVE=true

DRY_RUN=false
DRY_RUN=true

# Start processes in tmux session

SESSION=evaluate_reward_trajectory-$ID

tmux new-session -s $SESSION -n Xvnc -d
tmux new-window -t $SESSION:1 -n UE4 -c `pwd`
tmux new-window -t $SESSION:2 -n RosCore -c `pwd`
tmux new-window -t $SESSION:3 -n OctomapServerExt -c `pwd`
tmux new-window -t $SESSION:4 -n CollectData -c `pwd`
if $RUN_VISUALIZATION; then
  tmux new-window -t $SESSION:5 -n RViz -c `pwd`
  tmux new-window -t $SESSION:6 -n VncViewer -c `pwd`
fi

# Start VNC server
tmux send-keys -t $SESSION:0 "$XVNC_BINARY $VNC_DISPLAY -rfbport ${VNC_PORT} -geometry ${WINDOW_WIDTH}x${WINDOW_HEIGHT} -SecurityTypes None" C-m

# Start Unreal
#tmux send-keys -t $SESSION:1 "export DISPLAY=$VNC_DISPLAY" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxinfo" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxgears" C-m
if [ "$ENGINE_NAME" == "unreal" ]; then
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun $UE4EDITOR_CMD \\" C-m \
    "\"$SCENE_FILE\" \\" C-m \
    "\"$MAP\" \\" C-m \
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
  #tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun $BH_RENDERER_ZMQ_CMD " \
    "--width $WIDTH " \
    "--height $HEIGHT " \
    "--window-width $WINDOW_WIDTH " \
    "--window-height $WINDOW_HEIGHT " \
    "--mesh-filename $MESH_PATH " \
    "--use-msgpack-for-mesh true " \
    "--use-msgpack-for-mesh true " \
    "--model-scale $MODEL_SCALE " \
    "--model-yaw $MODEL_YAW " \
    "--model-pitch $MODEL_PITCH " \
    "--model-roll $MODEL_ROLL " \
    "--horz-fov $FOV " \
    "--address 'tcp://127.0.0.1:$BH_RENDERER_ZMQ_PORT' " \
    "--depth-scale 0.02; " \
    "sleep 1; done" C-m
else
  echo "Unknown engine: $ENGINE_NAME"
  exit 1
fi

# Roscore
tmux send-keys -t $SESSION:2 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:2 "roscore -p $ROSCORE_PORT" C-m

# Octomap server
tmux send-keys -t $SESSION:3 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:3 "sleep $OCTOMAP_WAIT_TIME" C-m
tmux send-keys -t $SESSION:3 "roslaunch octomap_server_ext $LAUNCH_FILE \\" C-m \
  "voxel_size:=$VOXEL_SIZE \\" C-m \
  "max_range:=$MAX_RANGE \\" C-m \
  "voxel_free_threshold:=$VOXEL_FREE_THRESHOLD \\" C-m \
  "voxel_occupied_threshold:=$VOXEL_OCCUPIED_THRESHOLD \\" C-m \
  "observation_count_saturation:=$OBSERVATION_COUNT_SATURATION \\" C-m \
  "point_cloud_filter_factor:=$POINT_CLOUD_FILTER_FACTOR " \
  "surface_voxels_filename:=\"$SURFACE_VOXELS_FILENAME\" \\" C-m \
  "binary_surface_voxels_filename:=\"$BINARY_SURFACE_VOXELS_FILENAME\" \\" C-m \
  "naive_scan_insertion:=\"$NAIVE_SCAN_INSERTION\" \\" C-m \
  "use_only_surface_voxels_for_score:=$USE_ONLY_SURFACE_VOXELS_FOR_SCORE \\" C-m \
  "octomap_server_ext_namespace:=octomap_server_ext_${ID}" C-m

if $RUN_VISUALIZATION; then
  # RViz
  tmux send-keys -t $SESSION:5 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
  tmux send-keys -t $SESSION:5 "sleep $OCTOMAP_WAIT_TIME" C-m
  tmux send-keys -t $SESSION:5 "rviz -d $RVIZ_CONFIG" C-m

  # VNCViewer
  tmux send-keys -t $SESSION:6 "sleep $OCTOMAP_WAIT_TIME" C-m
  tmux send-keys -t $SESSION:6 "vncviewer localhost:$VNC_PORT" C-m
fi


# Evaluation window
tmux send-keys -t $SESSION:4 "workon rl_reconstruct" C-m
tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:4 "sleep $EVALUATION_WAIT_TIME" C-m

echo "----- launch_evaluation_small_camera_trajectory.sh -----"
echo "id=$ID"
echo "scene_id=$SCENE_ID"
echo "camera_name=$CAMERA_NAME"
echo
echo "environment-config=$ENVIRONMENT_CONFIG"
echo "input_folder=$INPUT_FOLDER"
echo
echo "session=$SESSION$"

function run_evaluation {
  local INPUT_FILENAME_PREFIX=$1

  # Evaluation script
  tmux send-keys -t $SESSION:4 "python $HOME/rl_reconstruct/python/reward_learning/trajectory_playback.py \\" C-m \
      "--environment-config \"$ENVIRONMENT_CONFIG\" \\" C-m \
      "--client-id $ID \\" C-m \
      "--measure-timing $MEASURE_TIMING \\" C-m \
      "--input-filename-prefix $INPUT_FILENAME_PREFIX \\" C-m \
      "--episode 0 \\" C-m \
      "--max-steps 200 \\" C-m \
      "--verbose" C-m
}

#name="outputbuildings3_large_steps_200_stereo_model_0040_prediction_stereo_bm_0.50"
#run_evaluation $INPUT_FOLDER/${name}

#name="outputbuildings3_large_steps_200_stereo_model_0040_prediction_stereo_sgbm_0.50"
#run_evaluation $INPUT_FOLDER/${name}

##name="output20_steps_200_large_motion_model_0040_prediction"
#name="output30_steps_200_model_0040_prediction"
#run_evaluation $INPUT_FOLDER/${name}

#name="output20_steps_200_model_0040_prediction"
#name="output20_steps_200_detailed_model_0040_prediction"
#name="output20_steps_200_large_motion_model_0040_prediction"
#name="output30_steps_200_model_0040_prediction"
name="output30_steps_200_detailed_model_0040_prediction"
#name="output30_steps_200_32_model_0140_prediction"
#name="outputbuildings3_large_steps_200_model_0040_prediction"
#name="outputbuildings3_large_steps_200_detailed_model_0040_prediction"
#name="outputbuildings7_large_steps_200_model_0040_prediction"
#name="outputbuildings7_large_steps_200_detailed_model_0040_prediction"

#name="output20_steps_2000_model_0100_prediction"
#name="output30_steps_2000_model_0100_prediction"
#name="outputbuildings3_large_steps_2000_model_0100_prediction"
#name="outputbuildings7_large_steps_2000_model_0100_prediction"

name="output30_steps_200_detailed_random"

run_evaluation $INPUT_FOLDER/${name}

tmux attach -t $SESSION:4

