#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?Usage: bash run_infer.sh <root_dir> <save_dir>}"
SAVE_DIR="${2:?Usage: bash run_infer.sh <root_dir> <save_dir>}"

python infer_tfgridnet.py \
  --root_dir "$ROOT_DIR" \
  --split "test" \
  --ckpt "exp/tfgridnet/last.pt" \
  --save_dir "$SAVE_DIR" \
  --sample_rate 8000 \
  --num_speakers 2 \
  --n_channels 6 \
  --select_channels "0,2,4" \
  --n_samples 500 \
  --start_sample 0 \
  --compute_pesq \
  --compute_estoi \
  --strict_perceptual \
  --skip_existing \
  --verbose