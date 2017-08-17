#!/bin/bash

DATA_DIR=NONE
LOG_DIR=NONE
SCRIPT_DIR=NONE
MODEL_DIR=NONE

# Parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -d|--data-dir)
    DATA_DIR="$2"
    shift # pass argument
    ;;
    -p|--script-file-dir)
    SCRIPT_DIR="$2"
    shift # pass argument
    ;;
    -m|--model-dir)
    MODEL_DIR="$2"
    shift # pass argument
    ;;
    -l|--log-dir)
    LOG_DIR="$2"
  ;;
    *)
    echo Unkown option $key
    ;;
esac
shift # past argument or value
done

EXTRA_ARGS=$@

# Prints out the arguments that were passed into the script
echo "DATA_DIR=$DATA_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "EXTRA_ARGS=$EXTRA_ARGS"

export DATA_DIR
export LOG_DIR
export SCRIPT_DIR
export MODEL_DIR

DATA_PATH=$DATA_DIR
# DATA_PATH=$DATA_DIR/train
# TEST_DATA_PATH=$DATA_DIR/test
LOG_PATH=$LOG_DIR
MODEL_PATH=$MODEL_DIR

cp $0 $LOG_DIR/

# Add the root folder of the script to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

#pip install -e /var/storage/shared/intvc/benni/rl_reconstruct/python
pip install -e $SCRIPT_DIR/rl_reconstruct/python

python train.py \
    --data-path $DATA_PATH \
    --store-path $MODEL_PATH \
    --log-path $LOG_PATH \
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

