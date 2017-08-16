#!/bin/bash

UE4EDITOR_CMD=$HOME/UE4Editor
PROJECT_FILE="$HOME/Documents/Unreal Projects/CustomScenes/CustomScenes.uproject"

WIDTH=640
HEIGHT=480

#UNREALCV_PORT=9000
UNREALCV_PORT=$((9000+$1))

echo "Starting Unreal Engine with UnrealCV listening on port $UNREALCV_PORT"

$UE4EDITOR_CMD "$PROJECT_FILE" -game -WINDOWED -ResX=$WIDTH -ResY=$HEIGHT -UnrealCVPort $UNREALCV_PORT

