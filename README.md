# Multichannel-BSS-Diffusion
This repository is a **multichannel blind source separation (BSS)** project centered on a **diffusion prior** for speech, with a full pipeline that covers:

1) **Synthesizing a reproducible multichannel Mix2 dataset** (2 speakers, 6 channels; optional noise; with early/tail/observation splits)  
2) **Training and running diffusion-based multichannel BSS** (core method)  
3) **Running strong baselines** (AuxIVA / TF-GridNet / Neural-FCA) for comparison

---
##  Repository structure
```
.
├── README.md
├── baselines
│   ├── README.md
│   ├── iva
│   ├── neural_fca
│   └── tfgridnet
└── src
    ├── dataloaders
    ├── datasets
    ├── mc_bss_diffusion
    └── metrics
```
### `src/datasets` — Multichannel Mix2 dataset synthesis (2spk, 6ch)

This module provides a **fully scripted pipeline** to synthesize a multichannel dataset (layout inspired by SMS-WSJ style organization). It generates:

- metadata for LibriSpeech utterances
- metadata for WHAM noise
- mixture recipes (who mixes with whom, gains, optional noise)
- per-mixture reverberation/geometry parameters (room, mic array, sources, T60, etc.)
- rendered multi-channel audio using `pyroomacoustics`

Final audio is written in:

- `early/` (early reverberation)
- `tail/` (late reverberation tail)
- `observation/` (full mixture)

These are the testing (and training) datasets.

### `src/mc_bss_diffusion` — Core method: diffusion prior + multichannel BSS (train + inference)

This is the **core module** of the repository:

- **Training**: trains a diffusion prior on speech (via config-driven training)
- **Inference / separation**: runs diffusion posterior sampling (or equivalent sampling-based procedure) with **multichannel spatial post-processing** (e.g., FCP/IVA-like steps, depending on config/scripts)

This module is the main entrypoint for the diffusion-based multichannel separation experiments.

### `baselines` — Reference methods for comparison

Contains three baseline families, each with scripts to run separation and produce:

- separated waveforms (per-utterance folder outputs)
- `per_utt.csv` (utterance-level metrics)
- `summary.csv` (aggregate metrics)

Baselines included:

- **IVA (AuxIVA)**: classic blind baseline (no training)
- **TF-GridNet**: supervised baseline (train + infer)
- **Neural-FCA**: deep unsupervised baseline (train + infer)

### `src/metrics` — Evaluation utilities

Shared metric implementations / wrappers used by diffusion inference and/or baselines (e.g., SI-SDR/SDR, PESQ, eSTOI, etc., depending on how you run experiments).

### `src/dataloaders` — Data loading helpers

Dataset + loader utilities used by training and inference modules (e.g., loaders for prior training data).

------

## Recommended workflow (end-to-end)

### Step 1 — Synthesize Mix2 multichannel dataset (`src/datasets`)

Prepare:

- **LibriSpeech** 
- **WHAM noise** 

Run the pipeline (example; adjust paths/options to your setup):

```bash
bash src/datasets/run_pipeline.sh \
  --librispeech_dir "/path/to/LibriSpeech" \
  --wham_dir "/path/to/wham_noise" \
  --out_dir "/path/to/output/mix2_reverb_6ch" \
  --subsets "test-clean,train-clean-360" \
  --add_noise \
  --overwrite
```

Output (conceptually):

```text
/path/to/output/mix2_reverb_6ch/
  early/...
  tail/...
  observation/...
```

You will typically use `observation/` (or a chosen split) as the `root_dir` for inference/evaluation.

------

### Step 2 — Train the diffusion prior (`src/mc_bss_diffusion`)

Prepare:

- **LibriTTS** (commonly used for training a clean speech prior)

Update your config (YAML) to point to your LibriTTS root directory, then run the training script:

```bash
bash src/mc_bss_diffusion/train_prior.sh
```

Training outputs:

- experiment folder(s) under something like `exps/`
- checkpoints like `weights-*.pt`

(Optional) monitor training:

```bash
tensorboard --logdir exps --port 6006 --bind_all
```

------

### Step 3 — Run diffusion-based multichannel separation (`src/mc_bss_diffusion`)

After you have a checkpoint, run inference on the synthesized Mix2 dataset:

```bash
bash src/mc_bss_diffusion/run_infer.sh \
  --root_dir "/path/to/output/mix2_reverb_6ch" \
  --save_dir "/path/to/separation_outputs/mc_bss_diffusion" \
  --checkpoint "weights-139999.pt"
```

Notes:

- `root_dir` should point to the Mix2 dataset root (the script/config decides whether to use `observation/` and which split).
- `save_dir` will contain separated audio and metric CSVs (depending on script options).

------

### Step 4 — Run baselines for comparison (`baselines`)

#### 4.1 AuxIVA (no training)

```bash
bash baselines/iva/run_iva.sh \
  --root_dir "/path/to/output/mix2_reverb_6ch" \
  --save_dir "/path/to/separation_outputs/iva"
```

#### 4.2 TF-GridNet (supervised; train then infer)

Follow the baseline’s README/script in:

- `baselines/tfgridnet/`

#### 4.3 Neural-FCA (deep unsupervised; train then infer)

Follow the baseline’s README/script in:

- `baselines/neural_fca/`

All baselines are expected to produce:

- separated wavs (per-utterance)
- `per_utt.csv`
- `summary.csv`

