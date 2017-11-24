#!/bin/bash

export RLRECONSTRUCT_PY_DIR=$HOME/rl_reconstruct/python

DATA_BASE_PATH=$HOME/rl_reconstruct_datasets/reward_learning/small_camera
export DATA_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train.lmdb

HDF5_OUTPUT_PATH=$DATA_BASE_PATH/in_out_grid_16x16x16_0-1-2-3-4_EnvironmentNoPitch_CustomScenes_greedy_train_stats.hdf5

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/write_data_statistics_from_lmdb_to_hdf5.py \
  --lmdb-input-path $DATA_PATH \
  --hdf5-output-path $HDF5_OUTPUT_PATH

