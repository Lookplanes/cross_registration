#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data2/xujr/conda-envs/transmorph/bin/python}"
GPU="${GPU:-4}"
DATA_ROOT="${DATA_ROOT:-/data2/wuyh/HEMIT/dataset/processed_3717_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"
CHECKPOINT="${CHECKPOINT:-/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_fulltrain_100e_20260803/transcut_final.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg_data/hemit_stage2_offline_24k_v1}"

mkdir -p "${OUTPUT_DIR}"

# First formal scale: 12 translation directions x 2,000 = 24,000 train.
# Four disjoint source-ID shards write canonical, non-overlapping sample IDs.
# The parent shell waits for every GPU process before merging manifests.
GPU_LIST="${GPU_LIST:-4,5,6,7}"
IFS=',' read -r -a GPUS <<<"${GPU_LIST}"
DIRECTIONS=("0-1,0-2,0-3" "1-0,1-2,1-3" "2-0,2-1,2-3" "3-0,3-1,3-2")
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_LIST must contain exactly four comma-separated GPU IDs" >&2
  exit 2
fi

pids=()
for shard in 0 1 2 3; do
  echo "Starting train shard ${shard} on GPU ${GPUS[$shard]}: ${DIRECTIONS[$shard]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "${PYTHON}" -u \
    "${ROOT}/scripts/build_stage2_offline_dataset.py" \
    --checkpoint "${CHECKPOINT}" \
    --source-manifest "${MANIFEST_ROOT}/train_unpaired.csv" \
    --manifest-root "${DATA_ROOT}" \
    --source-split train \
    --output-split train \
    --output-dir "${OUTPUT_DIR}" \
    --directions "${DIRECTIONS[$shard]}" \
    --artifact-suffix "shard${shard}" \
    --samples-per-direction 2000 \
    --pair-direction alternating \
    --load-size 358 \
    --canvas-size 320 \
    --crop-size 256 \
    --batch-size 16 \
    --min-free-gb 30 \
    --device cuda \
    >"${OUTPUT_DIR}/build_train_shard${shard}.log" 2>&1 &
  pids+=("$!")
done

for shard in 0 1 2 3; do
  wait "${pids[$shard]}"
  echo "Train shard ${shard} complete"
done

"${PYTHON}" "${ROOT}/scripts/merge_stage2_offline_manifests.py" \
  --inputs \
    "${OUTPUT_DIR}/manifest_train_shard0.csv" \
    "${OUTPUT_DIR}/manifest_train_shard1.csv" \
    "${OUTPUT_DIR}/manifest_train_shard2.csv" \
    "${OUTPUT_DIR}/manifest_train_shard3.csv" \
  --root "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/manifest_train.csv" \
  --expected-samples 24000 \
  --expected-per-direction 2000 \
  --configs \
    "${OUTPUT_DIR}/dataset_config_train_shard0.json" \
    "${OUTPUT_DIR}/dataset_config_train_shard1.json" \
    "${OUTPUT_DIR}/dataset_config_train_shard2.json" \
    "${OUTPUT_DIR}/dataset_config_train_shard3.json" \
  --output-config "${OUTPUT_DIR}/dataset_config_train.json"

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
