#!/usr/bin/env bash
set -euo pipefail

# T13: one conditioned TransMorph jointly trained on all 12 HEMIT directions.
GPU="${GPU:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/stage2_conditioned_transmorph_t09_4domain_20e_20260802}"
DATA_ROOT="/data2/wuyh/HEMIT/dataset/processed_1200_150_150_512"
T09_CHECKPOINT="/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_50e_20260801/transcut_epoch_30.pth"

mkdir -p "${OUTPUT_DIR}/experiments"
exec >"${OUTPUT_DIR}/train.log" 2>&1

echo "Started: $(date '+%F %T %z')"
echo "GPU=${GPU}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" \
  /data2/xujr/conda-envs/transmorph/bin/python -u \
  scripts/train_synthetic_supervised.py \
  --transcut-ckpt "${T09_CHECKPOINT}" \
  --generator-ckpt "${T09_CHECKPOINT}" \
  --modality-dirs \
    "${DATA_ROOT}/train/he" \
    "${DATA_ROOT}/train/dapi" \
    "${DATA_ROOT}/train/panck" \
    "${DATA_ROOT}/train/cd3" \
  --val-modality-dirs \
    "${DATA_ROOT}/val/he" \
    "${DATA_ROOT}/val/dapi" \
    "${DATA_ROOT}/val/panck" \
    "${DATA_ROOT}/val/cd3" \
  --save-dir "${OUTPUT_DIR}" \
  --registration-model conditioned_transmorph \
  --img-size 256 256 \
  --batch-size 16 \
  --num-workers 4 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone-lr 1e-5 \
  --reg-weight 0.5 \
  --pair-direction random \
  --print-freq 100 \
  --val-interval 1 \
  --max-iters-per-epoch 0 \
  --max-val-iters 64 \
  --seed 42 \
  --device cuda

echo "Finished: $(date '+%F %T %z')"
