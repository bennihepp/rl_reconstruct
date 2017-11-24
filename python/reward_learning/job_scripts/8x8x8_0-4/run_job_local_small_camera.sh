#!/bin/bash

if [ -z $1 ]; then
    echo "Job id needs to be provided"
    exit 1
fi

JOB_ID=`printf "%04d" $1`

export CUDA_VISIBLE_DEVICES=$2

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

#DATA_BASE_PATH=$HOME/reward_learning/datasets/small_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/small_camera
#DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/small_camera_new
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/small_camera_big_scenes

#export DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train.lmdb
#export TEST_DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_test.lmdb
export DATA_PATH=$DATA_BASE_PATH/in_out_grid_8x8x8_0-1-2-3-4-5-6-7-8_greedy_train.lmdb
export TEST_DATA_PATH=$DATA_BASE_PATH/in_out_grid_8x8x8_0-1-2-3-4-5-6-7-8_greedy_test.lmdb
export DATA_TYPE=lmdb
#export MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera/${JOB_ID}_new
#export LOG_PATH=$HOME/rl_reconstruct_logs/reward_prediction_small_camera/${JOB_ID}_new
export MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera_big_scenes/${JOB_ID}
export LOG_PATH=$HOME/rl_reconstruct_logs/reward_prediction_small_camera_big_scenes/${JOB_ID}

mkdir -p $LOG_PATH
mkdir -p $MODEL_PATH

bash run_job_${JOB_ID}.sh ${@:3} --verbose-model 2>&1 | tee $LOG_PATH/stdout.log
