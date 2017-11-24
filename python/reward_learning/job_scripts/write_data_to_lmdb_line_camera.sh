#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

export RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

DATA_BASE_PATH=$HOME/reward_learning/datasets/line_camera/in_out_grid_16x16x16_0-1-2-3-4_HorizontalEnvironment_CustomScenes_buildings_greedy/

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_BASE_PATH=$DATA_BASE_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/write_data_to_lmdb.py \
  --config lmdb_config.yaml \
  --data-path $DATA_BASE_PATH/train \
  --lmdb-output-path $DATA_BASE_PATH/train_reward_prediction.lmdb \
  --test-data-path $DATA_BASE_PATH/test \
  --test-lmdb-output-path $DATA_BASE_PATH/test_reward_prediction.lmdb \
  --data.max_num_files -1 \
  --data.test_max_num_files -1 \
  --compute-stats true \
#  --only-compute-stats true

