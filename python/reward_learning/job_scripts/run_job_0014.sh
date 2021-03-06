#!/bin/bash

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOG_PATH=$LOG_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/train.py \
    --data-path $DATA_PATH \
    --store-path $MODEL_PATH \
    --log-path $LOG_PATH \
    \
    --max_num_train_files -1 \
    --max_num_test_files -1 \
    \
    --gpu_memory_fraction 0.5 \
    --intra_op_parallelism 4 \
    --inter_op_parallelism 4 \
    --cpu_train_queue_capacity $((1024 * 16)) \
    --cpu_test_queue_capacity $((1024 * 16)) \
    \
    --num_epochs 5000 \
    --batch_size 128 \
    --optimizer "adam" \
    --max_grad_global_norm "1e3" \
    --initial_learning_rate "1e-3" \
    --learning_rate_decay_epochs 25 \
    --learning_rate_decay_rate 0.96 \
    \
    --target_id "norm_rewards" \
    --input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --obs_levels_to_use "1,2" \
    --subvolume_slice_x "0,16" \
    --subvolume_slice_y "0,16" \
    --subvolume_slice_z "0,16" \
    \
    --num_convs_per_block 8 \
    --initial_num_filters 8 \
    --filter_increase_per_block 0 \
    --filter_increase_within_block 6 \
    --maxpool_after_each_block "false" \
    --max_num_blocks 1 \
    --max_output_grid_size -1 \
    --dropout_rate 0.3 \
    --add_biases_3dconv "false" \
    --activation_fn_3dconv "relu" \
    --num_units_regression "1024" \
    --activation_fn_regression "relu" \
    $@

    #--test-data-path $TEST_DATA_PATH
    #--learning_rate_decay_staircase
    #--add_biases_3dconv \
