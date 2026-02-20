# Baselines
This folder contains three baselines:
- **IVA (AuxIVA)**: blind source separation baseline.
- **TF-GridNet**: supervised baseline (implementation adapted from ESPnet).
- **Neural-FCA**: deep unsupervised baseline.

## IVA

### Run
From repo root (recommended), or from `baselines/` if your `iva_separator.py` already bootstraps `sys.path`.

```bash
bash baselines/run_iva.sh --root_dir "/path/to/test_dataset_root" --save_dir "/path/to/output_dir"
````

If you prefer calling Python directly:

```bash
python baselines/iva_separator.py \
  --root_dir "/path/to/test_dataset_root" \
  --split "test" \
  --save_dir "/path/to/output_dir" \
  --num_speakers 2 \
  --n_channels 6 \
  --select_channels "0,2,4" \
  --iva_iter 100 \
  --n_fft 2048 \
  --hop_length 256 \
  --win_length 2048 \
  --sep_ref_mic 0 \
  --metric_ref_mic 0 \
  --n_samples 500 \
  --start_sample 0 \
  --compute_pesq \
  --compute_estoi \
  --strict_perceptual \
  --skip_existing \
  --verbose
```

Outputs are written to `--save_dir/`:

* `per_utt.csv`
* `summary.csv`
* `<utt_id>/s1.wav, s2.wav, ...`

## TF-GridNet

### Train

```bash
bash baselines/tfgridnet/run_train.sh /path/to/train_dataset_root
```

Or call Python directly:

```bash
python baselines/tfgridnet/train_tfgridnet.py \
  --root_dir "/path/to/train_dataset_root" \
  --train_split "train" \
  --exp_dir "exp/tfgridnet" \
  --batch_size 5 \
  --num_workers 4 \
  --log_interval 10 \
  --max_len 16000 \
  --select_channels "0,2,4" \
  --n_src 2 \
  --sample_rate 8000 \
  --target_mode "early+late" \
  --ref_mic 0 \
  --epochs 60 \
  --lr 1e-3 \
  --grad_clip 1 \
  --n_fft 256 \
  --stride 64
```

### Inference + Evaluation

```bash
bash baselines/tfgridnet/run_infer.sh --root_dir "/path/to/test_dataset_root" --save_dir "/path/to/output_dir"
```

Or call Python directly:

```bash
python baselines/tfgridnet/infer_tfgridnet.py \
  --root_dir "/path/to/test_dataset_root" \
  --split "test" \
  --ckpt "exp/tfgridnet/last.pt" \
  --save_dir "/path/to/output_dir" \
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
```

Outputs are written to `--save_dir/`:

* `per_utt.csv`
* `summary.csv`
* `<utt_id>/s1.wav, s2.wav, ...`

## Reference

### TF-GridNet paper

Z. Wang, S. Cornell, S. Choi, Y. Lee, B. Kim, and S. Watanabe, "TF-GridNet: Making Time-Frequency Domain Models Great Again for Monaural Speaker Separation," *arXiv*, 2023, [Online]. Available: https://arxiv.org/abs/2209.03952.

### Code reference

TF-GridNet implementation is adapted based on ESPnet:

* [https://github.com/espnet/espnet](https://github.com/espnet/espnet)

