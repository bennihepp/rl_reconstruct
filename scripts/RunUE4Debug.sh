#!/bin/bash

UE4EDITOR_CMD=$HOME/UE4Editor
PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"
#MAP=/Game/maps/houses/medieval_building
MAP=/Game/maps/houses/medieval_building_with_input

WIDTH=640
HEIGHT=480
#WIDTH=800
#HEIGHT=600
FOV=120

UNREALCV_PORT=9000
#UNREALCV_PORT=$((9900+$1))

echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"

$UE4EDITOR_CMD "$PROJECT_FILE" $MAP -game -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -UnrealCVPort $UNREALCV_PORT -UnrealCVWidth $WIDTH -UnrealCVHeight $HEIGHT -UnrealCVFOV $FOV

