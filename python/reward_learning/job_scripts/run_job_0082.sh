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
    --data.max_num_train_files 50 \
    --data.max_num_test_files 50 \
    --data.fake_constant_data false \
    --data.fake_random_data false \
    --data.input_stats_filename ${DATA_PATH}/data_stats.hdf5 \
    \
    --io.validation_interval 10 \
    --io.train_summary_interval 10 \
    --io.model_summary_interval 10 \
    --io.model_summary_num_batches 1 \
    --io.checkpoint_int 50 \
    \
    --tf.gpu_memory_fraction 0.9 \
    --tf.intra_op_parallelism 4 \
    --tf.inter_op_parallelism 4 \
    --tf.cpu_train_queue_capacity $((1024 * 16)) \
    --tf.cpu_test_queue_capacity $((1024 * 16)) \
    --tf.cpu_train_queue_threads 4 \
    --tf.cpu_test_queue_threads 1 \
    --tf.log_device_placement false \
    --tf.create_tf_timeline false \
    \
    --training.num_epochs 200 \
    --training.batch_size 256 \
    --training.optimizer "adam" \
    --training.max_grad_global_norm "1e3" \
    --training.initial_learning_rate "1e-3" \
    --training.learning_rate_decay_epochs 25 \
    --training.learning_rate_decay_rate 0.96 \
    \
    --config config_0082.yaml \
    $@

    # --gpu_id "0,1,2,3" \
    #--keep_checkpoint_every_n_hours 2 \
    #--keep_n_last_checkpoints 5 \
    #--test-data-path $TEST_DATA_PATH \
    #--learning_rate_decay_staircase \
    #--add_biases_3dconv \
