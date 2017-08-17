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
    --validation_interval 10 \
    --train_summary_interval 10 \
    --model_summary_interval 10 \
    --model_summary_num_batches 25 \
    --checkpoint_interval 50 \
    \
    --gpu_memory_fraction 0.5 \
    --intra_op_parallelism 4 \
    --inter_op_parallelism 4 \
    --cpu_train_queue_capacity $((1024 * 16)) \
    --cpu_test_queue_capacity $((1024 * 16)) \
    --cpu_train_queue_threads 4 \
    --cpu_test_queue_threads 1 \
    \
    --num_epochs 500 \
    --batch_size 128 \
    --optimizer "adam" \
    --max_grad_global_norm "1e3" \
    --initial_learning_rate "1e-3" \
    --learning_rate_decay_epochs 25 \
    --learning_rate_decay_rate 0.96 \
    \
    --target_id "prob_rewards" \
    --input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --obs_levels_to_use "2" \
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
    --add_bias_3dconv "false" \
    --activation_fn_3dconv "relu" \
    --use_batch_norm_3dconv "true" \
    --num_units_regression "1024" \
    --activation_fn_regression "relu" \
    --use_batch_norm_regression "true" \
    $@

    #--keep_checkpoint_every_n_hours 2 \
    #--keep_n_last_checkpoints 5 \
    #--test-data-path $TEST_DATA_PATH \
    #--learning_rate_decay_staircase \
