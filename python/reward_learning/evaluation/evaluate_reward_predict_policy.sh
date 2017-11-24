#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/home/bhepp/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/train
# export DATA_PATH=/mnt/data1/reward_learning/datasets/in_out_grid_16x16x16_0-1-2-3_SimpleV2Environment_greedy
#TEST_DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment/

JOB_NUM=`printf "%04d" $1`
CHECKPOINT=$2

export MODEL_PATH=/home/bhepp/reward_learning/models/${JOB_NUM}/
#export MODEL_PATH=$HOME/models/reward_learning/1
# export MODEL_PATH=/mnt/data1/reward_learning/jobs/passed/models/cust-r-run_job_${JOB_NUM}.sh_rewlearn_v3\!~\!~\!4/models/
#export MODEL_PATH=/mnt/data1/reward_learning/jobs/running/models/cust-r-run_job_${JOB_NUM}.sh_rewlearn_v3\!~\!~\!8/models/

export SCRIPT_DIR="$(dirname $0)"
export RLRECONSTRUCT_PY_DIR=$SCRIPT_DIR/../..

#MODEL_CONFIG="model_config_${JOB_NUM}.yaml"
CONFIG="config_${JOB_NUM}.yaml"

echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
#echo "MODEL_CONFIG=$MODEL_CONFIG"
echo "CONFIG=$CONFIG"

ENVIRONMENT_CONFIG=../config/
SCORE_THRESHOLD=0.3
ENVIRONMENT=SimpleV2Environment
SCORE_THRESHOLD=0.4
ENVIRONMENT=HorizontalEnvironment
SCORE_THRESHOLD=0.6

USE_ORACLE=false
USE_ORACLE=true

#VISUALIZE=false
VISUALIZE=true

INTERACTIVE=false
#INTERACTIVE=true

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/evaluate_reward_predict_policy.py \
    --data-path "$DATA_PATH" \
    --model-path "$MODEL_PATH" \
    --config "$CONFIG" \
    --environment-config "$ENVIRONMENT_CONFIG" \
    --score-threshold $SCORE_THRESHOLD \
    --visualize $VISUALIZE \
    --interactive $INTERACTIVE \
    --use-oracle $USE_ORACLE \
    --verbose \
    \
    --tf.gpu_memory_fraction 0.5 \
    \
    --data.input_stats_filename ${DATA_PATH}/data_stats.hdf5

