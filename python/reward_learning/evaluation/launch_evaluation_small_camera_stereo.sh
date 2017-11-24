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

OUTPUT_FOLDER=outputs/output_${CAMERA_NAME}_new2
#OUTPUT_FOLDER=outputs/output_small_camera_vertical
# OUTPUT_FOLDER=output_small_camera_check
# OUTPUT_FOLDER=output_small_camera_tmp
#OUTPUT_FOLDER=output_rpg_ig
# OUTPUT_FOLDER=outputs/output_${CAMERA_NAME}

# New scenes
#SCENE_ID=20
#SCENE_ID=30
OUTPUT_FOLDER=outputs_scenes/output_${CAMERA_NAME}_fixed

if [ ! -d "$OUTPUT_FOLDER" ]; then
  echo "Output folder does not exist: $OUTPUT_FOLDER"
  exit 1
fi

LOG_FILE=$OUTPUT_FOLDER/log_${ID}_${SCENE_ID}.txt

RUN_RPG_IG_ACTIVE_RECONSTRUCTION=false
RUN_RPG_IG_ACTIVE_RECONSTRUCTION=true

RUN_VISUALIZATION=false
#RUN_VISUALIZATION=true

#export DATA_PATH=/home/bhepp/reward_learning/datasets/small_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/train
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}
#export DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train.lmdb
HDF5_DATA_STATS_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train_stats.hdf5

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

# SCENE_ID="1"
# SCENE_ID="2"
# SCENE_ID="3"
# SCENE_ID="3.1"
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/${CAMERA_NAME}_buildings${SCENE_ID}.yml
# ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/configs/small_camera_buildings${SCENE_ID}_EnvironmentNoPitch.yml

# New scenes
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes
HDF5_DATA_STATS_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_greedy_train_stats.hdf5
ENVIRONMENT_CONFIG=$HOME/rl_reconstruct/scenes/${CAMERA_NAME}_scene_${SCENE_ID}.yml

GPU_MEMORY_FRACTION=0.2

if [ -z "$GL_DISPLAY" ]; then
  GL_DISPLAY=:0
fi
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

  UE4=$HOME/src/UnrealEngine_4.17
  UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

  UNREALCV_PORT=$((9900 + $ID))

  WINDOW_WIDTH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.width --type int`
  WINDOW_HEIGHT=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.height --type int`
  FPS=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.fps --type int --default 20`
# Retrieve Unreal Engine parameters from YAML config
  SCENE_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_path`
  SCENE_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.scene_name`
  SCENE_FILE=$HOME/$SCENE_PATH/$SCENE_NAME/$SCENE_NAME.uproject
  MAP_PATH=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_path`
  MAP_NAME=`python -m pybh.tools.read_yaml_value $ENVIRONMENT_CONFIG unreal.map_name`
  MAP=$MAP_PATH/$MAP_NAME
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  OCTOMAP_WAIT_TIME=5
  EVALUATION_WAIT_TIME=20

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

NUM_EPISODES=50
MAX_FILE_NUM=100

MEASURE_TIMING=true

VISUALIZE=false
#VISUALIZE=true

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

SESSION=evaluate_reward_predict_policy_stereo-$ID

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
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun -d $GL_DISPLAY $UE4EDITOR_CMD \\" C-m \
    "\"$SCENE_FILE\" \\" C-m \
    "\"$MAP\" \\" C-m \
    "-game \\" C-m \
    "-FULLSCREEN \\" C-m \
    "-ResX=$WINDOW_WIDTH \\" C-m \
    "-ResY=$WINDOW_HEIGHT \\" C-m \
    "-VSync \\" C-m \
    "-UnrealCVPort $UNREALCV_PORT \\" C-m \
    "-UnrealCVWidth $WINDOW_WIDTH \\" C-m \
    "-UnrealCVHeight $WINDOW_HEIGHT \\" C-m \
    "-UnrealCVExitOnFailure true; \\" C-m \
    "sleep 1; done" C-m
  #tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD \"$SCENE_FILE\" \"$MAP\" -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m
elif [ "$ENGINE_NAME" == "bh_renderer_zmq" ]; then
  tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$VNC_DISPLAY vglrun -d $GL_DISPLAY $BH_RENDERER_ZMQ_CMD " \
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

if $RUN_RPG_IG_ACTIVE_RECONSTRUCTION; then
  FOCAL_LENGTH_X=`python -c "import math; import numpy as np; rad = $FOV * np.pi / 180.; print($WIDTH / (2 * math.tan(rad / 2)))"`
  FOCAL_LENGTH_Y=$FOCAL_LENGTH_X
  PRINCIPAL_POINT_X=`python -c "print(($WIDTH - 1) / 2.0)"`
  PRINCIPAL_POINT_Y=`python -c "print(($HEIGHT - 1) / 2.0)"`
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
  tmux send-keys -t $SESSION:7 "sleep $OCTOMAP_WAIT_TIME" C-m
  tmux send-keys -t $SESSION:7 "roslaunch ig_active_reconstruction_octomap octomap_world_representation_custom.launch \\" C-m \
    "voxel_size:=$VOXEL_SIZE \\" C-m \
    "max_range:=$MAX_RANGE \\" C-m \
    "occlusion_update_dist:=$OCCLUSION_UPDATE_DIST \\" C-m \
    "image_width:=$WIDTH \\" C-m \
    "image_height:=$HEIGHT \\" C-m \
    "focal_length_x:=$FOCAL_LENGTH_X \\" C-m \
    "focal_length_y:=$FOCAL_LENGTH_Y \\" C-m \
    "principal_point_x:=$PRINCIPAL_POINT_X \\" C-m \
    "principal_point_y:=$PRINCIPAL_POINT_Y \\" C-m \
    "voxels_in_void_ray:=$VOXELS_IN_VOID_RAY" C-m
fi

# Evaluation window
tmux send-keys -t $SESSION:4 "workon rl_reconstruct" C-m
tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  tmux send-keys -t $SESSION:4 "export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" C-m
fi
tmux send-keys -t $SESSION:4 "sleep $EVALUATION_WAIT_TIME" C-m

echo "----- launch_evaluation_small_camera_stereo.sh -----"
echo "id=$ID"
echo "scene_id=$SCENE_ID"
echo "camera_name=$CAMERA_NAME"
echo
echo "Using display $GL_DISPLAY for rendering"
echo
echo "environment-config=$ENVIRONMENT_CONFIG"
echo "output_folder=$OUTPUT_FOLDER"
echo
echo "plan_steps=$PLAN_STEPS"
echo "num_episodes=$NUM_EPISODES"
echo "max_file_num=$MAX_FILE_NUM"
echo "session=$SESSION$"

function run_evaluation {
  POLICY_MODE=$1

  if [ $POLICY_MODE == "prediction" ]; then
    MODEL_NUM=`printf "%04d" $2`
    CHECKPOINT=$3
    MODEL_POSTFIX=$4
    #OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}_model_${MODEL_NUM}${MODEL_POSTFIX}_25
    OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}_model_${MODEL_NUM}${MODEL_POSTFIX}
    #export MODEL_PATH=/home/bhepp/reward_learning/models/reward_prediction_small_camera/${MODEL_NUM}/
    MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera/${MODEL_NUM}${MODEL_POSTFIX}/
    MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera_big_scenes/${MODEL_NUM}${MODEL_POSTFIX}/
    CONFIG=$HOME/rl_reconstruct/python/reward_learning/job_scripts/config_${MODEL_NUM}.yaml
    echo "Preparing evaluation of policy $POLICY_MODE with model ${MODEL_NUM}${MODEL_POSTFIX}, checkpoint $CHECKPOINT"
  elif [ $POLICY_MODE == "action_prediction" ]; then
    MODEL_NUM=`printf "%04d" $2`
    CHECKPOINT=$3
    MODEL_POSTFIX=$4
    OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}_model_${MODEL_NUM}${MODEL_POSTFIX}
    MODEL_PATH=$HOME/rl_reconstruct_models/policy_a2c_small_camera/${MODEL_NUM}${MODEL_POSTFIX}/
    CONFIG=$HOME/rl_reconstruct/python/reward_learning/job_scripts/config_${MODEL_NUM}.yaml
    echo "Preparing evaluation of action policy $POLICY_MODE with model ${MODEL_NUM}${MODEL_POSTFIX}, checkpoint $CHECKPOINT"
  else
    OUTPUT_FILENAME_PREFIX=$OUTPUT_FOLDER/output${SCENE_ID}
    echo "Preparing evaluation of policy $POLICY_MODE"
  fi
  # OUTPUT_FILENAME_PREFIX=test_output3
  echo "  Output filename prefix: $OUTPUT_FILENAME_PREFIX"

  # Evaluation script
  tmux send-keys -t $SESSION:4 "python $HOME/rl_reconstruct/python/reward_learning/evaluate_reward_predict_policy_stereo.py \\" C-m \
      "--hdf5-data-stats-path \"$HDF5_DATA_STATS_PATH\" \\" C-m
  if [ $POLICY_MODE == "prediction" -o $POLICY_MODE == "action_prediction" ]; then
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
      "--prng-seed $PRNG_SEED \\" C-m \
      "--log-file $LOG_FILE \\" C-m \
      \
      "--tf.gpu_memory_fraction $GPU_MEMORY_FRACTION \\" C-m \
      \
      "--verbose" C-m
}

for i in {1..1}; do
#run_evaluation heuristic
#run_evaluation blind_oracle
#run_evaluation oracle
#run_evaluation uniform
#run_evaluation random

#run_evaluation prediction 5 model-64544
#run_evaluation prediction 6 model-44374
#run_evaluation prediction 7 model-165394
#run_evaluation prediction 8 model-44374
#run_evaluation prediction 9 model-64544
#run_evaluation prediction 10 model-84714
#run_evaluation prediction 18 model.ckpt-702780

# run_evaluation prediction 18 model.ckpt-1376900
#run_evaluation prediction 19 model.ckpt-1352400
#run_evaluation prediction 20 model.ckpt-1303400
#run_evaluation prediction 21 model.ckpt-960400
#run_evaluation prediction 22 model.ckpt-1470000
#run_evaluation prediction 22 model.ckpt-690900

#run_evaluation prediction 18 model.ckpt-1075500 _new
#run_evaluation oracle

#run_evaluation oracle
#run_evaluation blind_oracle

# run_evaluation rpg_ig_OcclusionAwareIg
# run_evaluation rpg_ig_UnobservedVoxelIg
# run_evaluation rpg_ig_AverageEntropyIg
# run_evaluation rpg_ig_RearSideVoxelIg
# run_evaluation rpg_ig_RearSideEntropyIg
#run_evaluation rpg_ig_ProximityCountIg
# run_evaluation rpg_ig_VasquezGomezAreaFactorIg

#run_evaluation action_prediction 18 model_dagger.ckpt-6762 _dagger_int_25_expl_0.1_2
#run_evaluation prediction 18 model.ckpt-1075500 _new
#run_evaluation prediction 18
run_evaluation prediction 40 model.ckpt-141949
#run_evaluation oracle
done

tmux attach -t $SESSION:4

