#!/usr/bin/env bash
set -euo pipefail

# Four-domain TransCUT run from random initialization with independently
# weighted GAN/NCE/identity losses and multi-scale structure preservation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data2/xujr/conda-envs/transmorph/bin/python}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_4modal_structure_20260715}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u scripts/train_transcut.py \
  --modality-config configs/translation/transcut_4modal_smoke.yaml \
  --split-manifest /data2/wuyh/bioimg_data/manifests/patches.csv \
  --manifest-root /data2/wuyh \
  --split train \
  --min-image-mean 5 \
  --min-image-std 5 \
  --max-dark-fraction 0.9 \
  --save-dir "${OUTPUT_DIR}" \
  --pairing-mode unpaired \
  --load-size 286 \
  --crop-size 256 \
  --embed-dim 48 \
  --ndf 16 \
  --num-patches 64 \
  --batch-size 1 \
  --num-workers 0 \
  --epochs 50 \
  --n-epochs-decay 50 \
  --lr 0.0002 \
  --lr-D 0.0001 \
  --d-update-freq 2 \
  --lambda-GAN 0.5 \
  --lambda-NCE 2.0 \
  --lambda-identity 1.0 \
  --lambda-structure 1.0 \
  --nce-fake-modality target \
  --seed 42 \
  --print-freq 100 \
  --sample-freq 2 \
  --sample-count 3 \
  --keep-sample-snapshots 6 \
  --save-epoch-freq 10 \
  --keep-epoch-checkpoints 3 \
  --device cuda \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
