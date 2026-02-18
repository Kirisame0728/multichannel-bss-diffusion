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







