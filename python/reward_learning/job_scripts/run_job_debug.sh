#!/bin/bash

DATA_DIR=NONE
LOG_DIR=NONE
SCRIPT_DIR=NONE
MODEL_DIR=NONE

# Parsing command line arguments:
while [[ $# > 1 ]]
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
    shift # pass argument
    ;;
    *)
    echo Unkown option $key
    shift # pass argument
    ;;
esac
shift # pass argument
done

EXTRA_ARG=$@

# Prints out the arguments that were passed into the script
echo "DATA_DIR=$DATA_DIR"
echo "LOG_DIR=$LOG_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "MODEL_DIR=$MODEL_DIR"
echo "EXTRA_ARG=$EXTRA_ARG"

export DATA_PATH=$DATA_DIR
# export DATA_PATH=$DATA_DIR/train
# export TEST_DATA_PATH=$DATA_DIR/test
export LOG_PATH=$LOG_DIR
export MODEL_PATH=$MODEL_DIR

export RLRECONSTRUCT_PY_DIR=$SCRIPT_DIR/../../

# Add the root folder of the script to the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$RLRECONSTRUCT_PY_DIR

export LC_ALL=C

#pip install -e /var/storage/shared/intvc/benni/rl_reconstruct/python
pip install -e $RLRECONSTRUCT_PY_DIR
pip install mpi4py

env

echo "Getting MPI rank"
exec $SCRIPT_DIR/mpi_rank.py
MPI_RANK=`exec $SCRIPT_DIR/mpi_rank.py`
echo "MPI_RANK=$MPI_RANK"

if [ -z "${MPI_RANK}" ]; then
    echo "Failed to query MPI rank"
    MPI_RANK=0
fi
