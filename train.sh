#!/bin/bash


set -o errexit
set -o pipefail
set -o nounset

if [[ -z "$CONDA_PREFIX" ]]; then
    conda activate ssl
fi

DATASET=$1
METHOD=$2


DATA_ROOT="./data"

CHECKPOINTS_ROOT="./outputs/checkpoints/"
mkdir -p $DATA_ROOT
mkdir -p $CHECKPOINTS_ROOT

DATASET_PATH="$DATA_ROOT"


CHECKPOINT_DIR="${CHECKPOINTS_ROOT}/train-${METHOD}-${DATASET}/"
mkdir -p $CHECKPOINT_DIR


python ./train.py \
    --backbone_arch resnet18 \
    --method $METHOD \
    --batch_size 128 \
    --optimizer sgd \
    --learning_rate_weights 0.05\
    --learning_rate_biases 0.05\
    --cosine \
    --checkpoint_dir $CHECKPOINT_DIR \
    --n_epochs 800 \
    $DATASET_PATH