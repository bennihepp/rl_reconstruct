#!/bin/bash

if [ -z $1 ]; then
    echo "Job id needs to be provided"
    exit 1
fi

JOB_ID=`printf "%04d" $1`

#if [ -n "$2" ]; then
#  export CUDA_VISIBLE_DEVICES=$2
#fi

MODEL_POSTFIX=$2

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

# Old scenes
#DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/small_camera
#DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train.lmdb
#TEST_DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_test.lmdb
#DATA_TYPE=lmdb
#MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera/${JOB_ID}${MODEL_POSTFIX}

# New scenes
CAMERA_NAME=small_camera
DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/${CAMERA_NAME}_big_scenes
DATA_TYPE=lmdb
DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_greedy_train.lmdb
TEST_DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_greedy_test.lmdb
HDF5_DATA_STATS_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_greedy_train_stats.hdf5
MODEL_PATH=$HOME/rl_reconstruct_models/reward_prediction_small_camera_big_scenes/${JOB_ID}${MODEL_POSTFIX}

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/evaluate_reward.py \
    --model-dir $MODEL_PATH \
    --shuffle true \
    --use-train-data false \
    --visualize false \
    \
    --data.train_path $DATA_PATH \
    --data.test_path $TEST_DATA_PATH \
    --data.type $DATA_TYPE \
    \
    --tf.gpu_memory_fraction 0.2 \
    \
    --config config_${JOB_ID}.yaml \
    ${@:3}

#    --hdf5-data-stats-path $HDF5_DATA_STATS_PATH \
