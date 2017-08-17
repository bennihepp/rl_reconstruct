#!/bin/bash

DATA_DIR="/hdfs/intvc/t-behepp/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment_V2"
LOG_DIR=tmp
MODEL_DIR=tmp
SCRIPT_DIR=.

export DATA_PATH=$DATA_DIR
# export DATA_PATH=$DATA_DIR/train
# export TEST_DATA_PATH=$DATA_DIR/test
export LOG_PATH=$LOG_DIR
export MODEL_PATH=$MODEL_DIR

export RLRECONSTRUCT_PY_DIR=$SCRIPT_DIR/../../

# Add the root folder of the script to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$RLRECONSTRUCT_PY_DIR

#pip install -e /var/storage/shared/intvc/benni/rl_reconstruct/python
pip install -e $RLRECONSTRUCT_PY_DIR

bash $@ --verbose --async_timeout 5
