#!/bin/bash

ID=$1

if [ -z "${ID}" ]; then
  echo "An id must be specified"
  exit 1
fi

export PYTHONPATH=$HOME/rl_reconstruct/python:$PYTHONPATH

SCENE_NAME=CustomScenes
MAP_NAME=buildings
ENVIRONMENT=HorizontalEnvironment
RESET_INTERVAL=500
RESET_SCORE_THRESHOLD=0.5
VOXEL_SIZE=0.2
MAX_RANGE=20.0
OBS_LEVELS="0,1,2,3,4"
OBS_SIZE="16"

OUTPUT_PATH="$HOME/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_${ENVIRONMENT}_${SCENE_NAME}_${MAP_NAME}_greedy/${ID}"
RECORDS_PER_FILE=100
NUM_FILES=1000

GL_DISPLAY=:0
DISPLAY=:$((1 + $ID))
WIDTH=640
HEIGHT=480
FPS=5

VNC_PORT=$((5900 + $ID))
UNREALCV_PORT=$((9900 + $ID))
ROSCORE_PORT=$((11911 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

UE4=/home/t-behepp/src/UnrealEngine_4.14
UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"

SCENE_PATH="$HOME/UnrealProjects/$SCENE_NAME"
PROJECT_FILE="$SCENE_PATH/$SCENE_NAME.uproject"
#MAP=/Game/maps/houses/medieval_building
MAP=/Game/maps/houses/medieval_building_with_input
MAP=/Game/maps/houses/$MAP_NAME

export ROS_MASTER_URI="http://localhost:$ROSCORE_PORT/"
SURFACE_VOXEL_FILENAME=$SCENE_PATH/Export/$MAP_NAME/surface_voxels.bin
LAUNCH_FILE=octomap_mapping_ext_rl.launch



SESSION=unreal-$ID

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
tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m
#tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m

tmux send-keys -t $SESSION:2 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:2 "roscore -p $ROSCORE_PORT" C-m
tmux send-keys -t $SESSION:3 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:3 "sleep 10" C-m
tmux send-keys -t $SESSION:3 "roslaunch octomap_server_ext $LAUNCH_FILE binary_surface_voxel_filename:=$SURFACE_VOXEL_FILENAME voxel_size:=$VOXEL_SIZE max_range:=$MAX_RANGE" C-m

tmux send-keys -t $SESSION:4 "export ROS_MASTER_URI='http://localhost:$ROSCORE_PORT/'" C-m
tmux send-keys -t $SESSION:4 "sleep 20" C-m
tmux send-keys -t $SESSION:4 "python $HOME/rl_reconstruct/python/reward_learning/collect_data.py --output-path $OUTPUT_PATH --obs-levels $OBS_LEVELS --obs-size $OBS_SIZE --environment $ENVIRONMENT --client-id $ID --records-per-file $RECORDS_PER_FILE --num-files $NUM_FILES --reset-interval $RESET_INTERVAL --reset-score-threshold $RESET_SCORE_THRESHOLD" C-m

tmux attach -t $SESSION:4

