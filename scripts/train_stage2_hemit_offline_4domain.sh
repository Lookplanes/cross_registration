#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data2/xujr/conda-envs/transmorph/bin/python}"
GPU="${GPU:-4}"
DATASET_DIR="${DATASET_DIR:-/data2/xujr/crossreg_data/hemit_stage2_offline_24k_v1}"
CHECKPOINT="${CHECKPOINT:-/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_fulltrain_100e_20260803/transcut_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/stage2_offline_fulltranscut_4domain_20e_v1}"

mkdir -p "${OUTPUT_DIR}"
exec >"${OUTPUT_DIR}/train.log" 2>&1

echo "Started: $(date '+%F %T %z')"
echo "GPU=${GPU}"
echo "DATASET_DIR=${DATASET_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u \
  "${ROOT}/scripts/train_stage2_offline.py" \
  --transcut-ckpt "${CHECKPOINT}" \
  --train-manifest "${DATASET_DIR}/manifest_train.csv" \
  --train-root "${DATASET_DIR}" \
  --val-manifest "${DATASET_DIR}/manifest_val.csv" \
  --val-root "${DATASET_DIR}" \
  --save-dir "${OUTPUT_DIR}" \
  --img-size 256 256 \
  --batch-size 16 \
  --num-workers 6 \
  --epochs 20 \
  --lr 1e-4 \
  --backbone-lr 1e-5 \
  --reg-weight 0.5 \
  --val-interval 1 \
  --val-samples-per-direction 1 \
  --val-sample-freq 1 \
  --keep-val-sample-snapshots 5 \
  --val-sample-flow-limit 15 \
  --milestone-freq 5 \
  --keep-milestones 3 \
  --device cuda

echo "Finished: $(date '+%F %T %z')"
