#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?Usage: bash run_train.sh <root_dir>}"

python train_neural_fca.py \
  --root_dir "$ROOT_DIR" \
  --train_split train \
  --exp_dir exp/neural-fca \
  --n_src 2 \
  --target_mode "early+late" \
  --select_channels 0,2,4 \
  --batch_size 4 \
  --num_workers 2 \
  --n_hiter 5 \
  --epochs 100 \
  --lr 5e-4 \
  --n_fft 512 \
  --hop 128 \
  --kl_cycle 10 \
  --kl_ratio 0.5 \
  --kl_max_beta 1.0 \
  --kl_max_beta_first 1.0 \
  --kl_first_epochs 10 \
  --limit 1000 \
  --max_len 16000 \
  --log_interval 20