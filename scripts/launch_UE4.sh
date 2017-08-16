#!/bin/bash

ID=$1

DISPLAY=:$((1 + $ID))
WIDTH=640
HEIGHT=480
FPS=5
VNC_PORT=$((5900 + $ID))
UNREALCV_PORT=$((9900 + $ID))

XVNC_BINARY=/opt/tigervnc-1.8.0/bin/Xvnc

UE4=/home/t-behepp/src/UnrealEngine_4.14
UE4EDITOR_CMD="$UE4/Engine/Binaries/Linux/UE4Editor"
PROJECT_FILE="/home/t-behepp/Documents/UnrealProjects/CustomScenes/CustomScenes.uproject"

#MAP=/Game/maps/houses/medieval_building
MAP=/Game/maps/houses/medieval_building_with_input
MAP=/Game/maps/houses/buildings

SESSION=unreal-$ID

tmux new-session -s $SESSION -n Xvnc -d
tmux new-window -t $SESSION:1 -n UE4 -c $HOME

# Start VNC server
tmux send-keys -t $SESSION:0 "$XVNC_BINARY $DISPLAY -rfbport ${VNC_PORT} -geometry ${WIDTH}x${HEIGHT} -SecurityTypes None" C-m

# Start Unreal
tmux send-keys -t $SESSION:1 "export DISPLAY=$DISPLAY" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxinfo" C-m
#tmux send-keys -t $SESSION:1 "vglrun glxgears" C-m
tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m
#tmux send-keys -t $SESSION:1 "while true; do DISPLAY=$DISPLAY vglrun $UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -FULLSCREEN -ResX=$WIDTH -ResY=$HEIGHT -VSync -FPS=$FPS -UnrealCVPort $UNREALCV_PORT; sleep 1; done" C-m

tmux attach -t $SESSION

