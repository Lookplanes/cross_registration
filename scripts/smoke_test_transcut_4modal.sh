#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/xujr/conda-envs/transmorph/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_smoke_4modal_scratch_20260714}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u scripts/train_transcut.py \
  --modality-config configs/translation/transcut_4modal_smoke.yaml \
  --split-manifest /data2/wuyh/bioimg_data/manifests/patches.csv \
  --manifest-root /data2/wuyh \
  --split train \
  --save-dir "${OUTPUT_DIR}" \
  --pairing-mode unpaired \
  --load-size 128 \
  --crop-size 128 \
  --embed-dim 32 \
  --ndf 16 \
  --num-patches 32 \
  --lr-D 0.0001 \
  --d-update-freq 2 \
  --batch-size 1 \
  --num-workers 0 \
  --epochs 1 \
  --n-epochs-decay 0 \
  --max-iters-per-epoch 40 \
  --print-freq 5 \
  --save-epoch-freq 1 \
  --device cuda \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
