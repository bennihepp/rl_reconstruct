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

OUTPUT_FOLDER=output
# OUTPUT_FOLDER=output_tmp
#OUTPUT_FOLDER=output_rpg_ig

RUN_RPG_IG_ACTIVE_RECONSTRUCTION=false
RUN_RPG_IG_ACTIVE_RECONSTRUCTION=true

RUN_VISUALIZATION=false
# RUN_VISUALIZATION=true

export DATA_PATH=/home/bhepp/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/train

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

# SCENE_ID="1"
# SCENE_ID="2"
# SCENE_ID="3"
# SCENE_ID="3.1"
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/line_camera_buildings${SCENE_ID}.yml

GPU_MEMORY_FRACTION=0.2

GL_DISPLAY=:0
VNC_DISPLAY=:$ID
WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.width --type int`
HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.height --type int`
FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.fps --type int --default 20`
UNREALCV_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.width --type int`
UNREALCV_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.height --type int`
UNREALCV_FOV=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG camera.fov --type float`

VNC_PORT=$((7000 + $ID))
UNREALCV_PORT=$((9900 + $ID))
ROSCORE_PORT=$((11911 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

UE4=/home/bhepp/src/UnrealEngine_4.17
UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

RVIZ_CONFIG="`dirname $0`/RLrecon.rviz"

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
POINT_CLOUD_FILTER_FACTOR=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.point_cloud_filter_factor --type float`
SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.surface_voxels_filename`
BINARY_SURFACE_VOXELS_FILENAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.binary_surface_voxels_filename`
NAIVE_SCAN_INSERTION=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.naive_scan_insertion --type bool`
USE_ONLY_SURFACE_VOXELS_FOR_SCORE=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG octomap.use_only_surface_voxels_for_score --type bool`

SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`

NUM_EPISODES=25
MAX_FILE_NUM=50

MEASURE_TIMING=true

VISUALIZE=false
# VISUALIZE=true

INTERACTIVE=false
# INTERACTIVE=true

IGNORE_COLLISION=false
# IGNORE_COLLISION=true

PLAN_STEPS=1
#PLAN_STEPS=2
#PLAN_STEPS=3

DRY_RUN=false
DRY_RUN=true

# Start processes in tmux session

SESSION=evaluate_reward_predict_policy-$ID

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
tmux send-keys -t $SESSION:0 "$XVNC_BINARY $VNC_DISPLAY -rfbport ${VNC_PORT} -geometry ${WIDTH}x${HEIGHT} -SecurityTypes None" C-m

# Start Unreal
#tmux send-keys -t $SESSION:1 "export DISPLAY=$VNC_DISPLAY" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxinfo" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxgears" C-m
tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun $UE4EDITOR_CMD \\" C-m \
  "\"$SCENE_FILE\" \\" C-m \
  " \"$MAP\" \\" C-m \
  "-game \\" C-m \
  "-FULLSCREEN \\" C-m \
  "-ResX=$WIDTH \\" C-m \
  "-ResY=$HEIGHT \\" C-m \
  "-VSync \\" C-m \
  "-UnrealCVPort $UNREALCV_PORT \\" C-m \
  "-UnrealCVWidth $UNREALCV_WIDTH \\" C-m \
  "-UnrealCVHeight $UNREALCV_HEIGHT; \\" C-m \
  "sleep 1; done" C-m
#tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m

# Roscore
tmux send-keys -t $SESSION:2 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:2 "roscore -p $ROSCORE_PORT" C-m

# Octomap server
tmux send-keys -t $SESSION:3 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:3 "sleep 10" C-m
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
  "use_only_surface_voxels_for_score:=$USE_ONLY_SURFACE_VOXELS_FOR_SCORE" C-m

if $RUN_VISUALIZATION; then
  # RViz
  tmux send-keys -t $SESSION:5 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
  tmux send-keys -t $SESSION:5 "sleep 10" C-m
  tmux send-keys -t $SESSION:5 "rviz -d $RVIZ_CONFIG" C-m

  # VNCViewer
  tmux send-keys -t $SESSION:6 "sleep 5" C-m
  tmux send-keys -t $SESSION:6 "vncviewer localhost:$VNC_PORT" C-m
fi

if $RUN_RPG_IG_ACTIVE_RECONSTRUCTION; then
  FOCAL_LENGTH_X=`python -c "import math; import numpy as np; rad = $UNREALCV_FOV * np.pi / 180.; print($UNREALCV_WIDTH / (2 * math.tan(rad / 2)))"`
  FOCAL_LENGTH_Y=$FOCAL_LENGTH_X
  PRINCIPAL_POINT_X=`python -c "print(($UNREALCV_WIDTH - 1) / 2.0)"`
  PRINCIPAL_POINT_Y=`python -c "print(($UNREALCV_HEIGHT - 1) / 2.0)"`
  VOXELS_IN_VOID_RAY=`python -c "print($MAX_RANGE / float($VOXEL_SIZE))"`
  # OCCLUSION_UPDATE_DIST=$MAX_RANGE
  OCCLUSION_UPDATE_DIST=`python -c "print($MAX_RANGE / 5.0)"`
  # echo $UNREALCV_WIDTH
  # echo $UNREALCV_HEIGHT
  # echo $FOCAL_LENGTH_X
  # echo $FOCAL_LENGTH_Y
  # echo $PRINCIPAL_POINT_X
  # echo $PRINCIPAL_POINT_Y
  # echo $VOXELS_IN_VOID_RAY
  # RPG IG active reconstruction Octomap server
  tmux new-window -t $SESSION:7 -n RPG_IG -c `pwd`
  tmux send-keys -t $SESSION:7 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
  tmux send-keys -t $SESSION:7 "sleep 10" C-m
  tmux send-keys -t $SESSION:7 "roslaunch ig_active_reconstruction_octomap octomap_world_representation_custom.launch \\" C-m \
    "voxel_size:=$VOXEL_SIZE \\" C-m \
    "max_range:=$MAX_RANGE \\" C-m \
    "occlusion_update_dist:=$OCCLUSION_UPDATE_DIST \\" C-m \
    "image_width:=$UNREALCV_WIDTH \\" C-m \
    "image_height:=$UNREALCV_HEIGHT \\" C-m \
    "focal_length_x:=$FOCAL_LENGTH_X \\" C-m \
    "focal_length_y:=$FOCAL_LENGTH_Y \\" C-m \
    "principal_point_x:=$PRINCIPAL_POINT_X \\" C-m \
    "principal_point_y:=$PRINCIPAL_POINT_Y \\" C-m \
    "voxels_in_void_ray:=$VOXELS_IN_VOID_RAY" C-m
fi

  # Evaluation window
  tmux send-keys -t $SESSION:4 "workon rl_reconstruct" C-m
  tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
  tmux send-keys -t $SESSION:4 "sleep 20" C-m

echo "----- launch_evaluation.sh -----"
echo "id=$ID"
echo "output_folder=$OUTPUT_FOLDER"
echo 
echo "plan_steps=$PLAN_STEPS"
echo "num_episodes=$NUM_EPISODES"
echo "max_file_num=$MAX_FILE_NUM"
echo "dry_run=$DRY_RUN"
echo "session=$SESSION$"

function run_evaluation {
  POLICY_MODE=$1

  if [ $POLICY_MODE == "prediction" ]; then
    MODEL_NUM=`printf "%04d" $2`
    CHECKPOINT=$3
    OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}_model_$MODEL_NUM
    export MODEL_PATH=/home/bhepp/reward_learning/models/${MODEL_NUM}/
    CONFIG=$HOME/rl_reconstruct/python/reward_learning/job_scripts/config_${MODEL_NUM}.yaml
    echo "Preparing evaluation of policy $POLICY_MODE with checkpoint $CHECKPOINT"
  else
    OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}
    echo "Preparing evaluation of policy $POLICY_MODE"
  fi
  # OUTPUT_FILENAME_PREFIX=test_output3
  echo "  Output filename prefix: $OUTPUT_FILENAME_PREFIX"

  # Evaluation script
  tmux send-keys -t $SESSION:4 "python $HOME/rl_reconstruct/python/reward_learning/evaluate_reward_predict_policy.py \\" C-m \
      "--data-path \"$DATA_PATH\" \\" C-m
  if [ $POLICY_MODE == "prediction" ]; then
    tmux send-keys -t $SESSION:4 " " \
        "--model-path \"$MODEL_PATH\" \\" C-m \
        "--config \"$CONFIG\" \\" C-m
  fi
  tmux send-keys -t $SESSION:4 " " \
      "--obs-levels \"$OBS_LEVELS\" \\" C-m \
      "--obs-sizes \"$OBS_SIZES\" \\" C-m \
      "--environment-config \"$ENVIRONMENT_CONFIG\" \\" C-m
  if [ -n "$CHECKPOINT" ]; then
      tmux send-keys -t $SESSION:4 " " "--checkpoint $CHECKPOINT \\" C-m
  fi
  tmux send-keys -t $SESSION:4 " " \
      "--client-id $ID \\" C-m \
      "--reset-interval $RESET_INTERVAL \\" C-m \
      "--reset-score-threshold $RESET_SCORE_THRESHOLD \\" C-m \
      "--visualize $VISUALIZE \\" C-m \
      "--measure-timing $MEASURE_TIMING \\" C-m \
      "--output-filename-prefix $OUTPUT_FILENAME_PREFIX \\" C-m \
      "--interactive $INTERACTIVE \\" C-m \
      "--ignore-collision $IGNORE_COLLISION \\" C-m \
      "--policy-mode $POLICY_MODE \\" C-m \
      "--num-episodes $NUM_EPISODES \\" C-m \
      "--plan-steps $PLAN_STEPS \\" C-m \
      "--dry-run $DRY_RUN \\" C-m \
      "--max-file-num $MAX_FILE_NUM \\" C-m \
      "--verbose \\" C-m \
      \
      "--tf.gpu_memory_fraction $GPU_MEMORY_FRACTION \\" C-m \
      \
      "--data.stats_filename ${DATA_PATH}/data_stats.hdf5 \\" C-m \
      "--data.normalize_input true \\" C-m \
      "--data.normalize_target false" C-m
}

# run_evaluation oracle
#run_evaluation blind_oracle
# run_evaluation prediction 7 model-165394
# run_evaluation prediction 8 model-44374
# run_evaluation prediction 9 model-64544
run_evaluation prediction 10 model-84714
run_evaluation blind_oracle
run_evaluation rpg_ig_ProximityCountIg

# run_evaluation prediction 5 model-64544
# run_evaluation prediction 6 model-44374
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg
# run_evaluation rpg_ig_RearSideVoxelIg
# run_evaluation prediction 5 model-64544
# run_evaluation prediction 6 model-44374
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg
# run_evaluation rpg_ig_RearSideVoxelIg


# run_evaluation prediction 5 model-64544
# run_evaluation prediction 6 model-44374
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg
# run_evaluation rpg_ig_RearSideVoxelIg
# run_evaluation prediction 5 model-64544
# run_evaluation prediction 6 model-44374
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg
# run_evaluation rpg_ig_RearSideVoxelIg

# run_evaluation rpg_ig_OcclusionAwareIg
# run_evaluation rpg_ig_UnobservedVoxelIg
# run_evaluation rpg_ig_AverageEntropyIg
# run_evaluation rpg_ig_RearSideVoxelIg
# run_evaluation rpg_ig_RearSideEntropyIg
# run_evaluation rpg_ig_ProximityCountIg
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg

#run_evaluation heuristic
# run_evaluation oracle
# run_evaluation blind_oracle
# run_evaluation uniform
# run_evaluation random
#run_evaluation prediction 5 model-64544
#run_evaluation prediction 6 model-44374
#run_evaluation prediction 7 model-165394
#run_evaluation prediction 8 model-44374
#run_evaluation prediction 9 model-64544
#run_evaluation prediction 10 model-84714

tmux attach -t $SESSION:4
