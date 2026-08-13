#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2/wuyh/HEMIT/dataset/processed_1200_150_150_512}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${DATA_ROOT}/manifests/signal05}"

# Four-domain supervised feasibility control. This does not replace the
# unpaired main experiment: it asks whether the existing shared architecture
# can learn aligned HEMIT mappings when the ambiguity is removed.
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_paired_upper_bound_20260801}"

exec "${ROOT}/scripts/train_transcut_hemit_4domain.sh" \
  --decoder-variant highres_content \
  --pairing-mode paired \
  --split-manifest "${MANIFEST_ROOT}/train_paired.csv" \
  --manifest-root "${DATA_ROOT}" \
  --split train \
  --fixed-sample-manifest "${MANIFEST_ROOT}/paired_test.csv" \
  --fixed-sample-root "${DATA_ROOT}" \
  --fixed-sample-split test \
  --lambda-paired 10.0 \
  --epochs 10 \
  --n-epochs-decay 0 \
  --sample-freq 1 \
  --sample-count 3 \
  --keep-sample-snapshots 5 \
  --milestone-freq 5 \
  --save-epoch-freq 1 \
  --keep-epoch-checkpoints 3 \
  "$@"
