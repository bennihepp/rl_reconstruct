#!/bin/bash

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "TEST_DATA_PATH=$TEST_DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOG_PATH=$LOG_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/train_v4.py \
    --data-path $DATA_PATH \
    --test-data-path $TEST_DATA_PATH \
    --store-path $MODEL_PATH \
    --log-path $LOG_PATH \
    --try-restore true \
    \
    --data.max_num_train_files -1 \
    --data.max_num_test_files -1 \
    --data.fake_constant_data false \
    --data.fake_random_data false \
    --data.stats_filename $DATA_PATH/data_stats.hdf5 \
    \
    --io.timeout 60 \
    --io.validation_interval 5 \
    --io.train_summary_interval 1 \
    --io.model_summary_interval 5 \
    --io.model_summary_num_batches 1 \
    --io.checkpoint_interval 5 \
    \
    --tf.gpu_memory_fraction 0.5 \
    --tf.intra_op_parallelism 4 \
    --tf.inter_op_parallelism 4 \
    --tf.cpu_train_queue_capacity $((1024 * 8)) \
    --tf.cpu_test_queue_capacity $((1024 * 8)) \
    --tf.cpu_train_queue_threads 4 \
    --tf.cpu_test_queue_threads 4 \
    --tf.cpu_train_queue_processes 4 \
    --tf.cpu_test_queue_processes 4 \
    --tf.log_device_placement false \
    --tf.create_tf_timeline false \
    \
    --training.num_epochs 200 \
    --training.batch_size 128 \
    --training.optimizer "adam" \
    --training.max_grad_global_norm "1e3" \
    --training.initial_learning_rate "2e-4" \
    --training.learning_rate_decay_epochs 15 \
    --training.learning_rate_decay_rate 0.96 \
    \
    --config config_0002.yaml \
    $@

    # --gpu_id "0,1,2,3" \
    #--keep_checkpoint_every_n_hours 2 \
    #--keep_n_last_checkpoints 5 \
    #--test-data-path $TEST_DATA_PATH \
    #--learning_rate_decay_staircase \
    #--add_biases_3dconv \
