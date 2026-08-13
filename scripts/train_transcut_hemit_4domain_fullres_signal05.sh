#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Geometry-preserving decoder ablation. All data, losses, DDP and retention
# settings are inherited from the maintained HEMIT launchers.
export DECODER_VARIANT="fullres_residual"
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_fullres_signal05_20260801}"

exec "${ROOT}/scripts/train_transcut_hemit_4domain_signal05.sh" \
  --epochs 50 \
  --n-epochs-decay 50 \
  --sample-freq 5 \
  --sample-count 3 \
  --keep-sample-snapshots 5 \
  --milestone-freq 10 \
  "$@"
