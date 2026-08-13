#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN="${TORCHRUN:-/data2/xujr/conda-envs/transmorph/bin/torchrun}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
NPROC="${NPROC:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_condneg_bs4_20260730}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT}"

# The 32 GB GPUs are substantially under-filled at batch size 1. A per-rank
# batch of 4 is the conservative high-throughput default for the next run;
# override these environment variables when benchmarking.
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Cross-modality mappings are not assumed to be invertible.  The experimental
# cycle-consistency path is therefore explicitly disabled in this main run.
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${TORCHRUN}" \
  --standalone \
  --nproc-per-node="${NPROC}" \
  scripts/train_transcut.py \
  --modality-config configs/translation/transcut_hemit_4domain_512.yaml \
  --save-dir "${OUTPUT_DIR}" \
  --pairing-mode unpaired \
  --min-image-mean 0 \
  --min-image-std 0 \
  --max-dark-fraction 1 \
  --load-size 286 \
  --crop-size 256 \
  --embed-dim 48 \
  --ndf 32 \
  --num-patches 64 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --prefetch-factor 2 \
  --epochs 50 \
  --n-epochs-decay 50 \
  --lr 0.0002 \
  --lr-D 0.0001 \
  --d-update-freq 1 \
  --lambda-GAN 1.0 \
  --lambda-NCE 1.0 \
  --lambda-identity 1.0 \
  --lambda-cycle 0.0 \
  --lambda-structure 0.0 \
  --lambda-D-mismatch 1.0 \
  --nce-fake-modality target \
  --seed 42 \
  --print-freq 100 \
  --sample-freq 1 \
  --sample-count 3 \
  --keep-sample-snapshots 5 \
  --milestone-freq 10 \
  --collapse-dark-gap 0.25 \
  --save-epoch-freq 1 \
  --keep-epoch-checkpoints 3 \
  --device cuda \
  "$@" >"${OUTPUT_DIR}/train.log" 2>&1
