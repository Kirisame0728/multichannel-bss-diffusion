#!/usr/bin/env bash
set -euo pipefail

python train_tfgridnet.py \
  --root_dir "/mnt/f/libri_train_360_2sp_6ch_8k" \
  --train_split "train" \
  --exp_dir "exp/tfgridnet" \
  --batch_size 4 \
  --num_workers 4 \
  --log_interval 10 \
  --max_len 32000 \
  --select_channels "0,2,4" \
  --n_src 2 \
  --sample_rate 8000 \
  --target_mode "early+late" \
  --ref_mic 0 \
  --epochs 30 \
  --lr 1e-3 \
  --grad_clip 1 \
  --n_fft 256 \
  --stride 64


