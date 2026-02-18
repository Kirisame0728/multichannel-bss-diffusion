# Mix2 (2sp-6ch) Reproducible Dataset Pipeline
This folder contains a fully reproducible pipeline to generate a **2-speaker, 6-channel** reverberant mixture dataset with an **SMS-WSJ-like** directory layout:

- `early/<split_dir>/...`  (early reflections per source)
- `tail/<split_dir>/...`   (late reflections per source)
- `observation/<split_dir>/...` (final multi-channel mixtures)

The pipeline is implemented by the following scripts:

- `create_libri_metadata.py` — create LibriSpeech utterance-level metadata
- `create_wham_metadata.py` — create WHAM noise metadata
- `create_mix_metadata.py` — create mixture “recipe” metadata (`mix2_*.csv`)
- `create_reverb_params.py` — sample room/mic/source geometry + T60 for each `mixture_ID`
- `create_mix_reverb_dataset.py` — render 6ch reverberant audio (early/tail/observation)

## 1. Directory Expectations
### 1.1 LibriSpeech
`--librispeech_dir` must be a LibriSpeech **root** directory containing subset folders such as:
```
LibriSpeech/
├── BOOKS.TXT
├── CHAPTERS.TXT
├── LICENSE.TXT
├── README.TXT
├── SPEAKERS.TXT
├── test-clean/
│   ├── 3570/
│   │   ├── 5694/
│   │   │   ├── 3570-5694-0000.flac
│   │   │   ├── 3570-5694-0001.flac
│   │   │   ├── 3570-5694-0002.flac
│   │   │   └── 3570-5694-0003.flac
│   │   └── 5695/
│   │       ├── 3570-5695-0000.flac
│   │       └── 3570-5695-0001.flac
│   ├── 3575/
│   │   └── 170457/
│   │       ├── 3575-170457-0000.flac
│   │       ├── 3575-170457-0052.flac
│   │       └── 3575-170457-0099.flac
│   └── 40/
│       └── 222/
│           ├── 40-222-0000.flac
│           └── 40-222-0001.flac
└── train-clean-360/
    ├── 176/
    │   ├── 122025/
    │   │   ├── 176-122025-0000.flac
    │   │   ├── 176-122025-0034.flac
    │   │   └── 176-122025-0100.flac
    │   └── 122026/
    │       ├── 176-122026-0000.flac
    │       └── 176-122026-0001.flac
    ├── 843/
    │   └── 101550/
    │       ├── 843-101550-0000.flac
    │       ├── 843-101550-0001.flac
    │       └── 843-101550-0010.flac
    └── 1673/
        └── 143396/
            ├── 1673-143396-0000.flac
            └── 1673-143396-0001.flac

```
### 1.2 WHAM noise

`--wham_dir` must point to the **WHAM noise root** directory. The scripts assume that noise files referenced by WHAM metadata are **relative to this root**.

A typical WHAM noise directory looks like:

```

wham_noise/
├── tr/                      # training noise
│   ├── noise/
│   │   ├── 00001.wav
│   │   ├── 00002.wav
│   │   └── ...
│   └── ...
├── cv/                      # development/validation noise
│   ├── noise/
│   │   ├── 00001.wav
│   │   └── ...
│   └── ...
└── tt/                      # test noise
├── noise/
│   ├── 00001.wav
│   └── ...
└── ...

```

> Notes  
> - The exact filenames under `tr/cv/tt` do not matter; what matters is that your WHAM metadata (`./metadata/wham/*.csv`) stores paths that can be resolved as:  
>   `abs_noise_path = wham_dir / origin_path`  
> - `create_mix_reverb_dataset.py` only reads noise if `--add_noise` is enabled (but the CLI still requires `--wham_dir`).


## 2. Virtual environment setup

### 2.1 Recommended (conda-forge)

```bash
conda create -n mixdata -c conda-forge -y \
  python=3.10 numpy pandas scipy soundfile tqdm pyloudnorm pyroomacoustics
conda activate mixdata

python -c "import pyroomacoustics as pra; print('pyroomacoustics ok')"
````

### 2.2 Minimal requirements.txt (pip)

This is the minimal Python-level dependency set needed by the pipeline:

```txt
numpy
pandas
soundfile
tqdm
pyloudnorm
scipy
pyroomacoustics
```

Install via:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
---

## 3. One-click generation via shell script

A one-click pipeline script is provided as `run_pipeline.sh`. It runs all stages in order:

1. LibriSpeech metadata → `./metadata/librispeech/*.csv`
2. WHAM metadata → `./metadata/wham/{train,dev,test}.csv`
3. mix2 recipe metadata → `./metadata/mix/mix2_*.csv`
4. reverb params for each mix2 split → `./reverb_params/mix/mix2_*.csv`
5. render dataset audio → `<out_dir>/{early,tail,observation}/<split_dir>/...`

### 3.1 Example usage

```bash
bash run_pipeline.sh \
  --librispeech_dir /path/to/LibriSpeech \
  --wham_dir /path/to/wham_noise/wham_noise \
  --out_dir /path/to/output/mix2_reverb_6ch \
  --subsets test-clean,train-clean-360 \
  --add_noise \
  --overwrite
```

### 3.2 Key knobs

* `--subsets`: which LibriSpeech subsets are used end-to-end (metadata + recipe + rendering)
* `--seed_mix`: controls recipe pairing/shuffling
* `--seed_reverb`: controls geometry/T60 sampling
* `--fs`: output sampling rate (default 8000)
* `--add_noise`: enable noise addition during rendering
* `--overwrite`: regenerate existing artifacts

---

## 4. Generation flow (script-by-script)

This section explains what each Python script generates, what it consumes, and where it writes outputs.

### 4.1 `create_libri_metadata.py` (LibriSpeech metadata)

**Input**

* `--librispeech_dir` (LibriSpeech root)
* In `--explicit` mode: `--test-clean DIR`, `--train-clean-360 DIR`, etc.

**Output**

* `./metadata/librispeech/<subset>.csv`

**Schema**

* `speaker_ID, sex, subset, length, origin_path`

**Important**

* `origin_path` must be relative to the LibriSpeech root, e.g.:

  * `test-clean/3575/170457/3575-170457-0052.flac`

---

### 4.2 `create_wham_metadata.py` (WHAM metadata)

**Input**

* `--wham_dir` (WHAM root)

**Output**

* `./metadata/wham/train.csv`
* `./metadata/wham/dev.csv`
* `./metadata/wham/test.csv`

**Schema (minimum)**

* `length, origin_path, augmented`

---

### 4.3 `create_mix_metadata.py` (mix2 recipe metadata)

This step creates the “recipe” describing which utterances are mixed, their gains, and optional noise.

**Input**

* `--librispeech_dir` (LibriSpeech root)
* `--librispeech_md_dir ./metadata/librispeech`
* `--wham_dir` and `--wham_md_dir ./metadata/wham`
* `--n_src 2`
* `--seed <seed_mix>`

**Output**

* `./metadata/mix/mix2_<split>.csv` (one per split discovered)

**Schema (required downstream)**

* `mixture_ID`
* `source_1_path, source_1_gain`
* `source_2_path, source_2_gain`
* (optional) `noise_path, noise_gain`

---

### 4.4 `create_reverb_params.py` (RIR parameter sampling)

This step generates the room/mic/source geometry and RT60 used for RIR simulation, **one row per mixture_ID**.

**Input**

* `--metadata_dir ./metadata/mix` (batch mode)
* Automatically scans: `mix2_*.csv`
* `--seed <seed_reverb>`

**Output**

* `./reverb_params/mix/mix2_<split>.csv` (same filenames as recipe files)

**Schema (required downstream)**

* `mixture_ID`
* room dimensions: `room_x, room_y, room_z`
* microphone positions: `mic1_x .. mic6_x, mic_y, mic_z` (linear array along x)
* source positions: `s1_x s1_y s1_z`, `s2_x s2_y s2_z`
* DOA (degrees): `s1_doa, s2_doa`
* reverberation: `T60`

#### 4.4.1 Sampling ranges (how parameters are drawn)

For each `mixture_ID`, the sampler draws (uniformly unless stated otherwise):

* **Room size**

  * `room_x ~ U(5, 10)`
  * `room_y ~ U(5, 10)`
  * `room_z ~ U(3, 4)`

* **Microphone array center**

  * placed near the room center:

    * `array_x = room_x/2 + U(-0.2, 0.2)`
    * `array_y = room_y/2 + U(-0.2, 0.2)`
  * `array_z ~ U(1, 2)`

* **Microphone geometry (6ch linear array)**

  * array is linear along x with spacing:

    * `mic_width = 0.05 m`
  * `mic_y` and `mic_z` are shared across all mics
  * `mic1_x .. mic6_x` are computed from center + offsets

* **Source directions / distances (2 speakers)**

  * DOA angles:

    * `theta ~ randint(0, 180)` (degrees)
  * Enforce a minimum separation:

    * `|theta1 - theta2| >= 15°` (rejection sampling)
  * Source distance:

    * `dist ~ U(0.75, 2.0) m`
  * Source positions are computed in the array plane (same z as array):

    * `sx = array_x + dist * cos(theta)`
    * `sy = array_y + dist * sin(theta)`
    * `sz = array_z`

* **Reverberation time (T60)**

  * depends on `reverb_level`:

    * `low`: `T60 ~ U(0.1, 0.3)`
    * `medium` (default): `T60 ~ U(0.2, 0.6)`
    * `high`: `T60 ~ U(0.4, 1.0)`

> The output of this stage is a deterministic function of `--seed` and the input CSV order.

---

### 4.5 `create_mix_reverb_dataset.py` (render 6ch early/tail/observation)

This is the final rendering stage. It reads the recipe + the matched params CSV, generates RIRs using `pyroomacoustics`, and writes multi-channel audio.

**Input**

* `--metadata_csv ./metadata/mix/mix2_<split>.csv`
* `--params_csv ./reverb_params/mix/mix2_<split>.csv`
* `--librispeech_dir` (for resolving `source_*_path`)
* `--wham_dir` (for resolving `noise_path` when `--add_noise`)
* `--out_dir <dataset_root>`
* `--split_dir <split_name>` (recommended: same as `<split>`)

**Output**

* `<out_dir>/early/<split_dir>/<mixture_ID>_0.wav`
* `<out_dir>/early/<split_dir>/<mixture_ID>_1.wav`
* `<out_dir>/tail/<split_dir>/<mixture_ID>_0.wav`
* `<out_dir>/tail/<split_dir>/<mixture_ID>_1.wav`
* `<out_dir>/observation/<split_dir>/<mixture_ID>.wav`

**Early vs tail**

* RIR is split into early and late parts by:

  * finding the direct-path onset around the RIR peak
  * keeping an early window of `--early_ms` (default 50ms)
  * tail = full - early

**Noise**

* enabled by `--add_noise`
* mono noise is repeated/cropped to match mixture length and added to all microphone channels
* additional attenuation controlled by `--noise_atten_db` (default 12 dB)

---

## 5. Quick checklist (before running)

1. LibriSpeech `origin_path` includes subset prefix (`test-clean/...`, `train-clean-360/...`)
2. WHAM metadata exists: `./metadata/wham/{train,dev,test}.csv`
3. mix2 recipe CSVs exist: `./metadata/mix/mix2_*.csv`
4. reverb params exist for each split: `./reverb_params/mix/mix2_*.csv`
5. `pyroomacoustics` imports successfully in your environment








