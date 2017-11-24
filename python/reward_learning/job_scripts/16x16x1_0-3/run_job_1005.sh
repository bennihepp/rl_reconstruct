#!/bin/bash

echo "RLRECONSTRUCT_PY_DIR=$RLRECONSTRUCT_PY_DIR"
echo "DATA_PATH=$DATA_PATH"
echo "TEST_DATA_PATH=$TEST_DATA_PATH"
echo "MODEL_PATH=$MODEL_PATH"
echo "LOG_PATH=$LOG_PATH"

python ${RLRECONSTRUCT_PY_DIR}/reward_learning/train.py \
    --model-dir $MODEL_PATH \
    --log-dir $LOG_PATH \
    --try-restore true \
    \
    --data.train_path $DATA_PATH \
    --data.test_path $TEST_DATA_PATH \
    --data.type $DATA_TYPE \
    --data.max_num_batches 1024 \
    --data.use_prefetching true \
    --data.prefetch_process_count 2 \
    --data.prefetch_queue_size 10 \
    --data.fake_constant_data false \
    --data.fake_random_data false \
    \
    --io.timeout 60 \
    --io.validation_interval 2 \
    --io.train_summary_interval 1 \
    --io.model_summary_interval 6 \
    --io.model_summary_num_batches 1 \
    --io.checkpoint_interval 6 \
    \
    --tf.gpu_memory_fraction 0.3 \
    --tf.intra_op_parallelism 4 \
    --tf.inter_op_parallelism 4 \
    --tf.sample_queue_capacity $((10 * 1024)) \
    --tf.sample_queue_min_after_dequeue $((5 * 1024)) \
    --tf.log_device_placement false \
    --tf.create_tf_timeline false \
    \
    --training.num_epochs 100 \
    --training.batch_size 128 \
    --training.optimizer "adam" \
    --training.max_grad_global_norm "1e3" \
    --training.initial_learning_rate "1e-3" \
    --training.learning_rate_decay_epochs 15 \
    --training.learning_rate_decay_rate 0.96 \
    \
    --config config_1005.yaml \
    $@

    # --gpu_id "0,1,2,3" \
    #--keep_checkpoint_every_n_hours 2 \
    #--keep_n_last_checkpoints 5 \
    #--test-data-path $TEST_DATA_PATH \
    #--learning_rate_decay_staircase \
    #--add_biases_3dconv \

