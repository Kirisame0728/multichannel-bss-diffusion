#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?Usage: bash run_infer_neural_fca.sh <root_dir> <save_dir>}"
SAVE_DIR="${2:?Usage: bash run_infer_neural_fca.sh <root_dir> <save_dir>}"

python infer_neural_fca.py \
  --root_dir "$ROOT_DIR" \
  --split test \
  --save_dir "$SAVE_DIR" \
  --ckpt exp/neural-fca/last.pt \
  --num_speakers 2 \
  --select_channels 0,2,4 \
  --n_samples 500 \
  --verbose \
  --compute_pesq \
  --compute_estoi \
  --n_fft 512 \
  --hop 128 \
  --n_iter 20 \
  --n_ziter 40 \
  --n_hiter 5