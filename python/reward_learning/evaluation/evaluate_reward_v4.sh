#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/home/bhepp/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/train

JOB_NUM=`printf "%04d" $1`
CHECKPOINT=$2

export MODEL_PATH=/home/bhepp/reward_learning/models/${JOB_NUM}/

RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

CONFIG=$HOME/rl_reconstruct/python/reward_learning/job_scripts/config_${JOB_NUM}.yaml

echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
#echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "CONFIG=$CONFIG"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/evaluate_reward_v4.py \
    --data-path "$DATA_PATH" \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --visualize true \
    --checkpoint "$CHECKPOINT" \
    \
    --tf.gpu_memory_fraction 0.3 \
    \
    --data.input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --data.normalize_input true \
    --data.normalize_target false
