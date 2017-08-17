#!/bin/bash

export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment/
#TEST_DATA_PATH=/mnt/data1/reward_learning/datasets/16x16x16_0-1-2-3_SimpleV0Environment/
MODEL_PATH=$HOME/models/reward_learning/1

python train.py \
    --data-path $DATA_PATH \
    --store-path $MODEL_PATH \
    \
    --max_num_train_files 1000000 \
    --max_num_test_files 1000000 \
    \
    --gpu_memory_fraction 0.5 \
    --intra_op_parallelism 4 \
    --inter_op_parallelism 4 \
    \
    --num_epochs 100000 \
    --batch_size 128 \
    --optimizer "adam" \
    --max_grad_global_norm "1e3" \
    --initial_learning_rate "1e-3" \
    --learning_rate_decay_epochs 25 \
    --learning_rate_decay_rate 0.96 \
    \
    --target_id "prob_rewards" \
    --input_stats_filename $MODEL_PATH/data_stats.hdf5 \
    --obs_levels_to_use "1,2" \
    --subvolume_slice_x "0,16" \
    --subvolume_slice_y "0,16" \
    --subvolume_slice_z "0,16" \
    \
    --num_filter_multiplier 8 \
    --max_output_grid_size 8 \
    --dropout_rate 0.3 \
    --activation_fn_3dconv "relu" \
    --num_units_regression "1024" \
    --activation_fn_regression "relu" \

    #--test-data-path $TEST_DATA_PATH
    #--learning_rate_decay_staircase
    #--add_biases_3dconv \

