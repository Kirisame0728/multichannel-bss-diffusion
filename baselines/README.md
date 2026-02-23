# Baselines
This folder contains three baselines:
- **IVA (AuxIVA)**: blind source separation baseline.
- **TF-GridNet**: supervised baseline (implementation adapted from ESPnet).
- **Neural-FCA**: deep unsupervised baseline.

## IVA

### Run
From repo root (recommended), or from `baselines/` if your `iva_separator.py` already bootstraps `sys.path`.

```bash
bash run_iva.sh --root_dir "/path/to/test_dataset_root" --save_dir "/path/to/output_dir"
````

If you prefer calling Python directly:

```bash
python iva_separator.py \
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
bash run_train.sh /path/to/train_dataset_root
```

Or call Python directly:

```bash
python train_tfgridnet.py \
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
bash run_infer.sh --root_dir "/path/to/test_dataset_root" --save_dir "/path/to/output_dir"
```

Or call Python directly:

```bash
python infer_tfgridnet.py \
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


## Neural-FCA

### Train
```bash
bash run_train.sh /path/to/train_dataset_root
```
Or call Python directly:
```bash
python train_neural_fca.py \
  --root_dir "/path/to/train_dataset_root" \
  --train_split train \
  --exp_dir exp/neural_fca \
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
```

### Inference + Evaluation
```bash
bash run_infer.sh --root_dir "/path/to/test_dataset_root" --save_dir "/path/to/output_dir"
```
Or call Python directly:
```bash
python infer_neural_fca.py \
  --root_dir /path/to/test_dataset_root" \
  --split test \
  --save_dir "/path/to/output_dir" \
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
```

## Reference

### TF-GridNet paper

Z.-Q. Wang, S. Cornell, S. Choi, Y. Lee, B.-Y. Kim, and S. Watanabe, “TF-GRIDNET: Making time-frequency domain models great again for monaural speaker separation,” in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2023, pp. 1–5, doi: 10.1109/ICASSP49357.2023.10094992. Available: https://arxiv.org/abs/2209.03952.

### Code reference

TF-GridNet implementation is adapted based on ESPnet:

* [https://github.com/espnet/espnet](https://github.com/espnet/espnet)

### Neural-FCA paper and code

Y. Bando, K. Sekiguchi, Y. Masuyama, A. A. Nugraha, M. Fontaine, and K. Yoshii, “Neural full-rank spatial covariance analysis for blind source separation,” IEEE Signal Processing Letters, vol. 28, pp. 1670–1674, 2021, doi: 10.1109/LSP.2021.3101699.

Code:

* [https://github.com/b-sigpro/neural-fca.spl2021](https://github.com/b-sigpro/neural-fca.spl2021)