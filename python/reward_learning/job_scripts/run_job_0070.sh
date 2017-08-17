#!/bin/bash

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOG_PATH=$LOG_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/train_v3.py \
    --data-path $DATA_PATH \
    --store-path $MODEL_PATH \
    --log-path $LOG_PATH \
    \
    --max_num_train_files 50 \
    --max_num_test_files 50 \
    --fake_constant_data false \
    --fake_random_data false \
    \
    --validation_interval 10 \
    --train_summary_interval 10 \
    --model_summary_interval 10 \
    --model_summary_num_batches 1 \
    --checkpoint_int 50 \
    \
    --gpu_memory_fraction 0.5 \
    --intra_op_parallelism 4 \
    --inter_op_parallelism 4 \
    --cpu_train_queue_capacity $((1024 * 16)) \
    --cpu_test_queue_capacity $((1024 * 16)) \
    --cpu_train_queue_threads 4 \
    --cpu_test_queue_threads 1 \
    --log_device_placement false \
    --create_tf_timeline false \
    \
    --num_epochs 500 \
    --batch_size 128 \
    --optimizer "adam" \
    --max_grad_global_norm "1e3" \
    --initial_learning_rate "1e-3" \
    --learning_rate_decay_epochs 25 \
    --learning_rate_decay_rate 0.96 \
    \
    --target_id "out_grid_3d" \
    --input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    --obs_levels_to_use "1,2" \
    --subvolume_slice_x "0,16" \
    --subvolume_slice_y "0,16" \
    --subvolume_slice_z "0,16" \
    --normalize_input true \
    --normalize_target false \
    \
    --model-config model_config_0060.yaml \
    $@

    # --gpu_id "0,1,2,3" \
    #--keep_checkpoint_every_n_hours 2 \
    #--keep_n_last_checkpoints 5 \
    #--test-data-path $TEST_DATA_PATH \
    #--learning_rate_decay_staircase \
    #--add_biases_3dconv \
