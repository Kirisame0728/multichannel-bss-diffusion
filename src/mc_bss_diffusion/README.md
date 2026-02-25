# Multichannel BSS Diffusion (mc_bss_diffusion)

This repository contains a multichannel blind source separation pipeline based on diffusion sampling and spatial post-processing (FCP/IVA). It supports:

- Training a diffusion model on LibriTTS-style speech segments.
- Inference (separation) on multichannel mixtures using a sampler with spatialization.
- Logging via TensorBoard.
- Checkpoint saving at fixed step intervals.

---

## 1. Repository layout
```
.
├── CQT_nsgt.py
├── FCP.py
├── IVA.py
├── README.md
├── conf
│   └── conf_libritts_unet1d_attention_8k.yaml
├── learner.py
├── model_parameters
│   └── ...
├── models
│   ├── unet_1d.py
│   ├── unet_1d_attn.py
│   └── utils.py
├── nsgt
│   ├── __init__.py
│   ├── audio.py
│   ├── cq.py
│   ├── fft.py
│   ├── fscale.py
│   ├── nsdual.py
│   ├── nsgfwin.py
│   ├── nsgfwin_sl.py
│   ├── nsgtf.py
│   ├── nsigtf.py
│   ├── plot.py
│   ├── reblock.py
│   ├── slicing.py
│   ├── slicq.py
│   ├── unslicing.py
│   └── util.py
├── requirements.txt
├── run_infer.sh
├── sampler.py
├── sampler_spatial_v1_reverb_iva_8kHz.py
├── sde.py
├── separate.py
├── stft.py
├── train.py
├── train_prior.sh
└── utils
    ├── logging.py
    └── setup.py
```
---

## 2. Environment setup

### 2.1 Create conda environment

```bash
conda create -n diffusion_bss_env python=3.10 -y
conda activate diffusion_bss_env
```
### 2.2 Install PyTorch + torchaudio

Pick ONE option that matches the machine.

#### CUDA (example: CUDA 12.1):
```bash
conda install -y pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
#### CPU-only:
```bash
conda install -y pytorch torchaudio cpuonly -c pytorch
```
### 2.3 Install Python dependencies
```bash
pip install -r requirements.txt
```
---
## 3. Dataset

Training uses `torchaudio.datasets.LIBRITTS` via `dataset_libritts` (wired inside `train.py` / `dataloaders/dataset_libritts.py` in the larger project, but here it is configured through Hydra YAML).
You must set the dataset root directory in the YAML:

- `dataset.root` should point to the directory that contains the `LibriTTS/` folder.

- The expected structure is typically:

  ```
  <dataset.root>/LibriTTS/train-clean-100/...
  <dataset.root>/LibriTTS/train-clean-360/...
  ...
  ```
You need to first download LibriTTS dataset in https://openslr.org/60/.

---

## 4. Training

### 4.1 Run training:
```bash
bash train_prior.sh
```
### 4.2 Training outputs
Training writes into `exps/`:

- `exps/<date-tag>/train.log`: stdout/stderr log
- `exps/<date-tag>/tb/`: TensorBoard event files
- `exps/raw_WAV_unet_att_8S_3S_8000hz/weights-*.pt`: model checkpoints

Checkpoint interval is controlled by the YAML / learner logic (commonly a fixed step interval such as every 20000 steps).
------
## 5. TensorBoard
If TensorBoard is installed:

```bash
tensorboard --logdir exps --port 6006 --bind_all
```

Open:

- `http://localhost:6006` (local)

Typical log path for one run:

- `exps/<date-tag>/tb`
-------
## 6. Inference (Separation)

### 6.1 Run inference

`run_test.sh` runs `separate.py` with CLI args.

```bash
bash run_test.sh
```

`separate.py` uses:

- `--config_path` to load YAML.
- `args.model_dir` + `args.inference.checkpoint` to locate weights (via `utils/setup.py::load_ema_weights`).
- 


### 6.2 Using a provided pretrained model (model_parameters)

If a pretrained package `model_parameters` is available:

1. Download `model_parameters` and extract it.
2. Place the extracted folder **directly under** `mc_bss_diffusion/`:
```
mc_bss_diffusion/
└── model_parameters/
    └── raw_WAV_unet_att_8S_3S_8000hz/
        ├── weights-xxxxx.pt
        └── ...
```
3. Edit `conf/conf_libritts_unet1d_attention_8k.yaml`:
Set `model_dir` to the extracted weights directory (relative paths are supported):
```yaml
model_dir: "model_parameters/raw_WAV_unet_att_8S_3S_8000hz"
```
and in `run_infer.sh`:
```shell
--checkpoint 'weights-139999.pt'
```

4. Ensure `run_test.sh` points to the correct YAML and uses a valid dataset root/save_dir.

### 6.3 Using checkpoints from training
If training produced checkpoints in:

```
exps/raw_WAV_unet_att_8S_3S_8000hz/weights-*.pt
```

Then set in the YAML: (line 9 in `conf_libritts_unet1d_attention_8k.yaml`)

```yaml
model_dir: "exps/raw_WAV_unet_att_8S_3S_8000hz"
```
and in `run_infer.sh`:
```shell
--checkpoint '<your-weights.pt>'
```


No code changes are required if `separate.py` is already loading via `model_dir + inference.checkpoint`.

### 6.4 Outputs

Inference writes outputs into the specified `save_dir`, organized by:

- number of speakers
- reverb/anechoic mode
- architecture
- blind/oracle mode
- and a timestamp tag

------
## 7. Typical workflow

1. Prepare LibriTTS dataset under the configured `dataset.root`.

2. Install environment (`diffusion_bss_env`) and dependencies.

3. Train:

   ```bash
   bash train_prior.sh
   ```

4. Monitor:

   ```bash
   tensorboard --logdir exps --port 6006 --bind_all
   ```

5. Run inference:

   ```bash
   bash run_test.sh
   ```

------
## 8. File/Module description

### Top-level scripts

- `train.py`
  Entry point for training. Loads Hydra config, builds dataset + model, instantiates `Learner`, and starts training.
- `learner.py`
  Training loop implementation: optimizer steps, EMA weight handling, periodic sampling, checkpointing, and TensorBoard logging.
- `separate.py`
  Entry point for inference. Loads a trained model checkpoint (EMA weights), loads evaluation dataset, runs the sampler, performs spatialization, and writes separation outputs + metrics.
- `train_prior.sh`
  Convenience wrapper to launch training with environment variables (CUDA device selection, thread caps, Hydra full error).
- `run_infer.sh`
  Convenience wrapper to launch inference with a fixed set of CLI arguments.

### Core signal processing / spatial modules

- `FCP.py`
  Full-rank spatial covariance processing utilities. Includes routines used by the sampler spatialization and SNR calculations.
- `IVA.py`
  Independent Vector Analysis implementation for multichannel separation initialization / baseline routines.
- `stft.py`
  STFT/ISTFT helpers used by the sampler and/or spatial processing.
- `CQT_nsgt.py`
  CQT wrapper built on top of `nsgt/` modules, used for CQT-based visualization or transforms.
- `nsgt/`
  NSGT implementation details (frequency scales, window generation, slicing/unslicing, etc.) used by `CQT_nsgt.py`.

### Diffusion / sampling

- `sde.py`
  SDE definition(s) used by the diffusion model and sampler (e.g., VE SDE / elucidating settings).
- `sampler.py`
  Core diffusion sampling loop utilities (predict, unconditional sampling, step scheduling, etc.).
- `sampler_spatial_v1_reverb_iva_8kHz.py`
  Main separation sampler that integrates diffusion sampling with spatialization (FCP/IVA) for multichannel separation.

### Models

- `models/unet_1d.py`
  A 1D UNet architecture (baseline).
- `models/unet_1d_attn.py`
  Attention-augmented 1D UNet architecture used by the current training config (recommended).
- `models/utils.py`
  Shared model utilities (building blocks, normalization, helper layers).

### Utilities

- `utils/setup.py`
  Checkpoint loading helpers (e.g., loading EMA weights into a model).
- `utils/logging.py`
  Logging and visualization helpers (TensorBoard summaries, waveform/spectrogram/CQT plotting hooks, etc.).

### Config

- `conf/conf_libritts_unet1d_attention_8k.yaml`
  Main Hydra config: dataset path, model architecture selection, training hyperparameters, EMA, checkpoint interval, TensorBoard directory, inference defaults.

### Experiments

- `exps/<date-tag>/`
  A single run folder with:
  - `tb/`: TensorBoard event files
  - `train.log`: console logs
- `exps/raw_WAV_unet_att_8S_3S_8000hz/`
  Checkpoint storage folder (weights saved periodically).

## 9. Credits

**This project is based on ArrayDPS:**
