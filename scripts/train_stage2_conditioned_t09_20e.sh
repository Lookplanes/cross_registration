#!/usr/bin/env bash
set -euo pipefail

# T12: original TransMorph topology with pair-modality conditioning.
# Override GPU or OUTPUT_DIR from the environment when needed.
GPU="${GPU:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/stage2_conditioned_transmorph_t09_e30_20e_20260802}"
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
  --data-dir /tmp/stage2_probe_hemit/train \
  --val-data-dir /tmp/stage2_probe_hemit/val \
  --save-dir "${OUTPUT_DIR}" \
  --registration-model conditioned_transmorph \
  --src-modality 0 \
  --tgt-modality 1 \
  --img-size 256 256 \
  --batch-size 4 \
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
