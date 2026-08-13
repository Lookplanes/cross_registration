#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2/wuyh/HEMIT/dataset/processed_1200_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"
DECODER_VARIANT="${DECODER_VARIANT:-highres_content}"

# Isolated data-distribution ablation:
# - model, losses and all four domains are unchanged;
# - training is still unpaired;
# - H&E/DAPI use all tissue patches;
# - panCK/CD3 independently require >=5% signal;
# - exported references use aligned test rows where all three markers meet 5%.
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_highres_signal05_20260731}"

exec "${ROOT}/scripts/train_transcut_hemit_4domain.sh" \
  --decoder-variant "${DECODER_VARIANT}" \
  --split-manifest "${MANIFEST_ROOT}/train_unpaired.csv" \
  --manifest-root "${DATA_ROOT}" \
  --split train \
  --fixed-sample-manifest "${MANIFEST_ROOT}/paired_test.csv" \
  --fixed-sample-root "${DATA_ROOT}" \
  --fixed-sample-split test \
  "$@"
