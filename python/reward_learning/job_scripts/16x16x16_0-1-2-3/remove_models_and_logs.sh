#!/bin/bash

if [ -z $1 ]; then
    echo "Job id needs to be provided"
    exit 1
fi

JOB_ID=`printf "%04d" $1`

#CAMERA_NAME=line_camera
CAMERA_NAME=small_camera

export MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_${CAMERA_NAME}_big_scenes/16x16x16_0-1-2-3/$JOB_ID
export LOG_PATH=$HOME/rl_reconstruct_logs/reward_prediction_${CAMERA_NAME}_big_scenes/16x16x16_0-1-2-3/$JOB_ID

echo "Are you sure you want to delete models in ${MODEL_PATH} and logs in ${LOG_PATH}? [y/n] "

read confirmation

if [ "$confirmation" == "y" ]; then
  echo rm -r $LOG_PATH
  rm -r $LOG_PATH
  echo rm -r $MODEL_PATH
  rm -r $MODEL_PATH
else
  echo "Canceled"
fi

