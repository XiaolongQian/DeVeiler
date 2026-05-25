#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DEBUG_MAX_ITER="${DEBUG_MAX_ITER:-2}"
export DEBUG_SKIP_VAL="${DEBUG_SKIP_VAL:-1}"

python basicsr/train.py -opt options/train/DeVeiler/stage1_pretrain.yml --debug
python basicsr/train.py -opt options/train/DeVeiler/stage2_reblur.yml --debug
python basicsr/train.py -opt options/train/DeVeiler/stage3_finetune.yml --debug
