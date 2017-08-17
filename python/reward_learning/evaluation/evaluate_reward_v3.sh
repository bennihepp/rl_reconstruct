#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/mnt/data1/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_SimpleV0Environment_greedy
export DATA_PATH=/mnt/data1/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_SimpleV2Environment_greedy
#TEST_DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment/

JOB_NUM=`printf "%04d" $1`
CHECKPOINT=$2

#export MODEL_PATH=$HOME/models/reward_learning/1
export MODEL_PATH=/mnt/data1/reward_learning/jobs/passed/models/cust-r-run_job_${JOB_NUM}.sh_rewlearn_v3\!~\!~\!4/models/
#export MODEL_PATH=/mnt/data1/reward_learning/jobs/running/models/cust-r-run_job_${JOB_NUM}.sh_rewlearn_v3\!~\!~\!8/models/

export SCRIPT_DIR="$(dirname $0)"
export RLRECONSTRUCT_PY_DIR=$SCRIPT_DIR/../..

CONFIG="config_${JOB_NUM}.yaml"
#MODEL_CONFIG="model_config_${JOB_NUM}.yaml"

echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
#echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "CONFIG=$CONFIG"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/evaluate_reward_v3.py \
    --data-path "$DATA_PATH" \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --visualize true \
    \
    --tf.gpu_memory_fraction 0.5 \
    \
    --data.input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --data.normalize_input true \
    --data.normalize_target false

