#!/bin/bash

if [ -z $1 ]; then
    echo "Job id needs to be provided"
    exit 1
fi

JOB_ID=`printf "%04d" $1`

export CUDA_VISIBLE_DEVICES=$2

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

CAMERA_NAME=line_camera
DATA_SIZE="32x32x1_0-1-2"
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes

export DATA_PATH=$DATA_BASE_PATH/in_out_grid_${DATA_SIZE}_greedy_train.lmdb
export TEST_DATA_PATH=$DATA_BASE_PATH/in_out_grid_${DATA_SIZE}_greedy_test.lmdb
export DATA_TYPE=lmdb
export MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_${CAMERA_NAME}_big_scenes/${JOB_ID}
export LOG_PATH=$HOME/rl_reconstruct_logs/reward_prediction_${CAMERA_NAME}_big_scenes/${JOB_ID}

mkdir -p $LOG_PATH
mkdir -p $MODEL_PATH

bash run_job_${JOB_ID}.sh ${@:3} --verbose-model 2>&1 | tee $LOG_PATH/stdout.log

