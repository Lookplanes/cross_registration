#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2/wuyh/HEMIT/dataset/processed_1200_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"

# Main distribution remains unpaired. Only 5% of sampling events draw from a
# fixed, deterministically selected 5% subset of eligible aligned train patches.
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_semipaired05_50e_20260801}"

exec "${ROOT}/scripts/train_transcut_hemit_4domain.sh" \
  --decoder-variant highres_content \
  --pairing-mode unpaired \
  --split-manifest "${MANIFEST_ROOT}/train_unpaired.csv" \
  --manifest-root "${DATA_ROOT}" \
  --split train \
  --paired-anchor-manifest "${MANIFEST_ROOT}/train_paired_anchor_05.csv" \
  --paired-anchor-probability 0.05 \
  --fixed-sample-manifest "${MANIFEST_ROOT}/paired_test.csv" \
  --fixed-sample-root "${DATA_ROOT}" \
  --fixed-sample-split test \
  --lambda-paired 10.0 \
  --epochs 25 \
  --n-epochs-decay 25 \
  --sample-freq 5 \
  --sample-count 3 \
  --keep-sample-snapshots 5 \
  --milestone-freq 10 \
  --save-epoch-freq 1 \
  --keep-epoch-checkpoints 3 \
  "$@"
