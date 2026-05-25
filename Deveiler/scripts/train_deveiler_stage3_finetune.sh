#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python basicsr/train.py -opt options/train/DeVeiler/stage3_finetune.yml
