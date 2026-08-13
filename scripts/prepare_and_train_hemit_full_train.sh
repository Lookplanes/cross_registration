#!/usr/bin/env bash
set -euo pipefail

# Expand the already-downloaded HEMIT archive to the complete official train
# split, rebuild the four-domain 512 px patch dataset, and then launch the
# maintained semi-paired TransCUT recipe.  Every stage is restart-safe:
# extraction keeps existing files, while completed preprocessing is reused.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEMIT_ROOT="${HEMIT_ROOT:-/data2/wuyh/HEMIT}"
ARCHIVE="${ARCHIVE:-${HEMIT_ROOT}/temp/HEMIT.7z}"
ARCHIVE_LIST="${ARCHIVE_LIST:-${HEMIT_ROOT}/temp/logs/archive-list.txt}"
RAW_ROOT="${RAW_ROOT:-${HEMIT_ROOT}/dataset/sample}"
OLD_MANIFEST="${OLD_MANIFEST:-${HEMIT_ROOT}/dataset/manifests/expanded_pairs_1200_150_150.csv}"
EXPANDED_MANIFEST="${EXPANDED_MANIFEST:-${HEMIT_ROOT}/dataset/manifests/expanded_pairs_3717_150_150.csv}"
NEW_PATHS="${NEW_PATHS:-${HEMIT_ROOT}/temp/expand_3717_150_150_new_paths.txt}"
NEW_PATTERNS="${NEW_PATTERNS:-${HEMIT_ROOT}/temp/expand_3717_150_150_new_patterns.txt}"
DATA_ROOT="${DATA_ROOT:-${HEMIT_ROOT}/dataset/processed_3717_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"
OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_fulltrain_100e_20260803}"
BSDTAR="${BSDTAR:-/data2/xujr/miniconda3/bin/bsdtar}"
PYTHON="${PYTHON:-/data2/xujr/conda-envs/transmorph/bin/python}"

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/pipeline.log") 2>&1

echo "[$(date '+%F %T %z')] pipeline start"
df -h /data2

"${PYTHON}" "${HEMIT_ROOT}/scripts/expand_hemit_sample.py" \
  --archive-list "${ARCHIVE_LIST}" \
  --existing-manifest "${OLD_MANIFEST}" \
  --output-manifest "${EXPANDED_MANIFEST}" \
  --new-paths "${NEW_PATHS}" \
  --new-patterns "${NEW_PATTERNS}" \
  --train 3717 --val 150 --test 150 \
  --seed 20260729-incremental-v1

echo "[$(date '+%F %T %z')] extracting missing TIFF files"
"${BSDTAR}" -xkf "${ARCHIVE}" -C "${RAW_ROOT}" -T "${NEW_PATTERNS}"

for spec in train:3717 val:150 test:150; do
  split="${spec%%:*}"
  expected="${spec##*:}"
  input_count="$(find "${RAW_ROOT}/${split}/input" -maxdepth 1 -type f -name '*.tif' | wc -l)"
  label_count="$(find "${RAW_ROOT}/${split}/label" -maxdepth 1 -type f -name '*.tif' | wc -l)"
  echo "${split}: input=${input_count} label=${label_count} expected=${expected}"
  if [[ "${input_count}" -ne "${expected}" || "${label_count}" -ne "${expected}" ]]; then
    echo "ERROR: extracted pair count mismatch for ${split}" >&2
    exit 20
  fi
done

if [[ ! -f "${DATA_ROOT}/manifests/patches.csv" ]]; then
  if [[ -e "${DATA_ROOT}" || -e "${DATA_ROOT}.incomplete" ]]; then
    echo "ERROR: incomplete or invalid preprocessing output already exists" >&2
    exit 21
  fi
  echo "[$(date '+%F %T %z')] preprocessing aligned four-domain patches"
  "${PYTHON}" "${HEMIT_ROOT}/scripts/preprocess_hemit_offline.py" \
    --input-root "${RAW_ROOT}" \
    --output-root "${DATA_ROOT}" \
    --patch-size 512 \
    --stride 512 \
    --min-tissue-fraction 0.15 \
    --tissue-od-threshold 0.08 \
    --marker-signal-threshold 5
else
  echo "[$(date '+%F %T %z')] reusing completed preprocessing output ${DATA_ROOT}"
fi

echo "[$(date '+%F %T %z')] building signal-filtered training manifests"
"${PYTHON}" "${ROOT}/scripts/prepare_hemit_signal_manifests.py" \
  --input-manifest "${DATA_ROOT}/manifests/patches.csv" \
  --output-dir "${MANIFEST_ROOT}" \
  --signal-threshold 0.05 \
  --paired-anchor-fraction 0.05 \
  --seed 42

echo "[$(date '+%F %T %z')] GPU status before training"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
df -h /data2

echo "[$(date '+%F %T %z')] launching 4-GPU TransCUT"
cd "${ROOT}"
DATA_ROOT="${DATA_ROOT}" \
MANIFEST_ROOT="${MANIFEST_ROOT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
GPU_IDS="${GPU_IDS:-4,5,6,7}" \
NPROC="${NPROC:-4}" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
NUM_WORKERS="${NUM_WORKERS:-6}" \
  exec "${ROOT}/scripts/train_transcut_hemit_4domain_semipaired05.sh" \
    --modality-config configs/translation/transcut_hemit_4domain_fulltrain_512.yaml \
    --epochs 50 \
    --n-epochs-decay 50 \
    --sample-freq 5 \
    --milestone-freq 10 \
    --keep-epoch-checkpoints 3
