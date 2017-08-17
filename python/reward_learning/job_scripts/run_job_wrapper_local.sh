#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment_V2/
#TEST_DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment/
export MODEL_PATH=$HOME/models/reward_learning/1
export LOG_PATH=$MODEL_PATH

export SCRIPT_DIR="$(dirname $0)"
export RLRECONSTRUCT_PY_DIR=$SCRIPT_DIR/../..

JOBSCRIPT=`printf "run_job_%04d.sh" $1`
echo "JOBSCRIPT=${JOBSCRIPT}"

bash $SCRIPT_DIR/$JOBSCRIPT --verbose --async_timeout 5
