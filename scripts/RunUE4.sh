#!/bin/bash

UE4EDITOR_CMD=$HOME/UE4Editor
PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"
MAP=/Game/maps/houses/medieval_building
MAP=/Game/maps/houses/buildings

WIDTH=640
HEIGHT=480
#WIDTH=320
#HEIGHT=240

#UNREALCV_PORT=9000
UNREALCV_PORT=$((9900+$1))

echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"

$UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -UnrealCVPort $UNREALCV_PORT

