#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data2/xujr/conda-envs/transmorph/bin/python}"
GPU="${GPU:-4}"
DATA_ROOT="${DATA_ROOT:-/data2/wuyh/HEMIT/dataset/processed_3717_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"
CHECKPOINT="${CHECKPOINT:-/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_fulltrain_100e_20260803/transcut_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg_data/hemit_stage2_offline_60k_v1}"

mkdir -p "${OUTPUT_DIR}"

# 12 translation directions x 5000 = 60,000 balanced training pairs.
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
  "${ROOT}/scripts/build_stage2_offline_dataset.py" \
  --checkpoint "${CHECKPOINT}" \
  --source-manifest "${MANIFEST_ROOT}/train_unpaired.csv" \
  --manifest-root "${DATA_ROOT}" \
  --source-split train \
  --output-split train \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-direction 5000 \
  --pair-direction alternating \
  --load-size 358 \
  --canvas-size 320 \
  --crop-size 256 \
  --batch-size 16 \
  --min-free-gb 30 \
  --device cuda

# Fixed held-out construction: 12 directions x 64 = 768 validation pairs.
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
  "${ROOT}/scripts/build_stage2_offline_dataset.py" \
  --checkpoint "${CHECKPOINT}" \
  --source-manifest "${MANIFEST_ROOT}/paired_test.csv" \
  --manifest-root "${DATA_ROOT}" \
  --source-split test \
  --output-split val \
  --output-dir "${OUTPUT_DIR}" \
  --samples-per-direction 64 \
  --pair-direction alternating \
  --load-size 358 \
  --canvas-size 320 \
  --crop-size 256 \
  --batch-size 16 \
  --min-free-gb 30 \
  --device cuda

echo "Offline Stage 2 dataset complete: ${OUTPUT_DIR}"
