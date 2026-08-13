#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# This is an isolated architecture experiment. The legacy HEMIT launcher and
# its checkpoints remain unchanged and can be selected simply by using
# train_transcut_hemit_4domain.sh directly.
export OUTPUT_DIR="${OUTPUT_DIR:-/data2/xujr/crossreg/transcut_hemit_4domain_highres_20260731}"
export BATCH_SIZE="${BATCH_SIZE:-4}"

exec "${ROOT}/scripts/train_transcut_hemit_4domain.sh" \
  --decoder-variant highres_content \
  "$@"
