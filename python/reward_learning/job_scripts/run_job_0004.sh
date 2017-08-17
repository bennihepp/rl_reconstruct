#!/bin/bash

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOG_PATH=$LOG_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/train.py \
    --data-path $DATA_PATH \
    --test-data-path $DATA_PATH \
    --store-path $MODEL_PATH \
    --log-path $LOG_PATH \
    \
    --max_num_train_files 100 \
    --max_num_test_files 100 \
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
    --learning_rate_decay_staircase "false" \
    \
    --target_id "mean_occupancy" \
    --input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --obs_levels_to_use "1,2" \
    --subvolume_slice_x "0,16" \
    --subvolume_slice_y "0,16" \
    --subvolume_slice_z "0,16" \
    \
    --num_convs_per_block 2 \
    --initial_num_filters 8 \
    --filter_increase_per_block 8 \
    --filter_increase_within_block 0 \
    --maxpool_after_each_block "true" \
    --max_num_blocks -1 \
    --max_output_grid_size 8 \
    --dropout_rate 0.3 \
    --add_biases_3dconv "false" \
    --activation_fn_3dconv "relu" \
    --num_units_regression "1024" \
    --activation_fn_regression "relu" \

    #--test-data-path $TEST_DATA_PATH
