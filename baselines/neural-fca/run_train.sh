#!/usr/bin/env bash
set -euo pipefail

python train_neural_fca.py \
  --root_dir "/mnt/f/libri_train_360_2sp_6ch_8k" \
  --train_split train \
  --exp_dir exp/neural-fca \
  --n_src 2 \
  --target_mode "early+late" \
  --select_channels 0,2,4 \
  --batch_size 4 \
  --num_workers 2 \
  --n_hiter 5 \
  --epochs 60 \
  --lr 1e-3 \
  --n_fft 512 \
  --hop 128 \
  --kl_cycle 10 \
  --kl_ratio 0.5 \
  --kl_max_beta 1.0 \
  --kl_max_beta_first 10.0 \
  --kl_first_epochs 50 \
  --limit 1000 \
  --max_len 16000 \
  --log_interval 20