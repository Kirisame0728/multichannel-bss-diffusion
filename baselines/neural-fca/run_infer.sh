python baselines/neural-fca/infer_neural_fca.py \
  --root_dir /mnt/d/datasets/your_mix2_root \
  --split test \
  --save_dir results/neural-fca/run1 \
  --ckpt baselines/neural-fca/exp/run1/last.pt \
  --num_speakers 2 \
  --select_channels 0,2,4 \
  --n_samples 500 \
  --verbose