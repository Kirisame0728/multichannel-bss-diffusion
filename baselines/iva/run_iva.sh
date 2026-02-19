#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?Usage: bash run.sh <root_dir> <save_dir>}"
SAVE_DIR="${2:?Usage: bash run.sh <root_dir> <save_dir>}"

python iva_separator.py \
  --root_dir "$ROOT_DIR" \
  --dataset mclibri2mix \
  --split test \
  --num_speakers 2 \
  --iva_iter 100 \
  --n_fft 2048 \
  --hop_length 256 \
  --win_length 2048 \
  --ref_mic 0 \
  --force_smswsj_3ch \
  --save_dir "$SAVE_DIR" \
  --n_samples 500 \
  --start_sample 0 \
  --skip_existing

