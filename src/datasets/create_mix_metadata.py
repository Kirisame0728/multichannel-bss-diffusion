#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


EPS = 1e-10
MAX_AMP = 0.9
RATE = 16000
MIN_LOUDNESS = -33.0
MAX_LOUDNESS = -25.0


@dataclass(frozen=True)
class Config:
    librispeech_dir: Path
    librispeech_md_dir: Path
    wham_dir: Path
    wham_md_dir: Path
    metadata_outdir: Path
    n_src: int
    seed: int
    dev_test_mixtures: int
    round_to: int
    overwrite: bool
    strict: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create mix metadata (recipe) from LibriSpeech + WHAM metadata.")
    p.add_argument("--librispeech_dir", type=str, required=True, help="LibriSpeech root directory")
    p.add_argument("--librispeech_md_dir", type=str, required=True, help="Directory containing LibriSpeech CSV metadata")
    p.add_argument("--wham_dir", type=str, required=True, help="WHAM noise root directory")
    p.add_argument("--wham_md_dir", type=str, required=True, help="Directory containing WHAM CSV metadata")
    p.add_argument("--metadata_outdir", type=str, default=None, help="Output directory for mix metadata CSVs")
    p.add_argument("--n_src", type=int, required=True, help="Number of sources in each mixture (e.g. 2)")
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--dev_test_mixtures", type=int, default=3000)
    p.add_argument("--round_to", type=int, default=100, help="Round number of rows down to a multiple of this (0 disables).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p


def split_from_librispeech_filename(name: str) -> str:
    # Expected: train-clean-360.csv / test-clean.csv / dev-clean.csv ...
    # We follow the original behavior: use prefix before '-'
    return name.split("-")[0]


def find_wham_file(wham_files: Sequence[str], split: str) -> str:
    # Original behavior: first WHAM csv whose filename starts with split
    for f in wham_files:
        if f.startswith(split) and f.endswith(".csv"):
            return f
    raise FileNotFoundError(f"No WHAM metadata CSV found for split='{split}' in wham_md_dir.")


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, engine="python")


def maybe_round(df: pd.DataFrame, round_to: int) -> pd.DataFrame:
    if round_to and round_to > 0:
        n = (len(df) // round_to) * round_to
        return df.iloc[:n].reset_index(drop=True)
    return df.reset_index(drop=True)


def ensure_outdir(cfg: Config) -> None:
    cfg.metadata_outdir.mkdir(parents=True, exist_ok=True)


def resolve_outdir(librispeech_dir: Path, metadata_outdir: Optional[str]) -> Path:
    if metadata_outdir is not None:
        return Path(metadata_outdir).expanduser().resolve()
    # Keep original default style, just rename directory from LibriMix to Mix
    root = librispeech_dir.parent
    return (root / "Mix" / "metadata").resolve()


def validate_file_exists(p: Path, strict: bool, what: str) -> None:
    if p.exists():
        return
    msg = f"{what} does not exist: {p}"
    if strict:
        raise FileNotFoundError(msg)
    print(f"[WARN] {msg}")


def load_audio_mono(path: Path, strict: bool) -> np.ndarray:
    try:
        wav, _ = sf.read(str(path), dtype="float32")
    except Exception as e:
        if strict:
            raise
        raise RuntimeError(f"Failed to read audio: {path} ({e})") from e
    if wav.ndim > 1:
        wav = wav[:, 0]
    return wav


def pad_or_trim(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == length:
        return x
    if len(x) < length:
        return np.pad(x, (0, length - len(x)), mode="constant")
    return x[:length]


def choose_utterance_pairs_train(md: pd.DataFrame, n_src: int, rng: random.Random) -> List[List[int]]:
    # Greedy: shuffle indices, then pick groups with distinct speakers.
    # Goal: each utterance used at most once (train preference).
    indices = list(range(len(md)))
    rng.shuffle(indices)

    pairs: List[List[int]] = []
    used = set()

    # Map speaker -> list of indices (shuffled) to allow distinct speaker sampling faster.
    speaker_to_indices: Dict[str, List[int]] = {}
    for idx in indices:
        spk = str(md.iloc[idx]["speaker_ID"])
        speaker_to_indices.setdefault(spk, []).append(idx)

    speakers = list(speaker_to_indices.keys())
    rng.shuffle(speakers)

    # Use a round-robin over speakers to reduce collisions.
    # This is not guaranteed optimal but is fast and respects "distinct speakers".
    speaker_ptr = {s: 0 for s in speakers}

    def pop_next(s: str) -> Optional[int]:
        p = speaker_ptr[s]
        lst = speaker_to_indices[s]
        if p >= len(lst):
            return None
        speaker_ptr[s] = p + 1
        return lst[p]

    # Build pairs
    active_speakers = [s for s in speakers if speaker_ptr[s] < len(speaker_to_indices[s])]
    while len(active_speakers) >= n_src:
        rng.shuffle(active_speakers)
        chosen_speakers = active_speakers[:n_src]
        chosen_indices: List[int] = []
        ok = True
        for s in chosen_speakers:
            idx = pop_next(s)
            if idx is None or idx in used:
                ok = False
                break
            chosen_indices.append(idx)
        if not ok:
            # refresh active_speakers and continue
            active_speakers = [s for s in speakers if speaker_ptr[s] < len(speaker_to_indices[s])]
            continue
        for idx in chosen_indices:
            used.add(idx)
        pairs.append(chosen_indices)
        active_speakers = [s for s in speakers if speaker_ptr[s] < len(speaker_to_indices[s])]

    return pairs


def choose_utterance_pairs_devtest(md: pd.DataFrame, n_src: int, target: int, rng: random.Random) -> List[List[int]]:
    # Sample until we have `target` unique mixtures (unique by set of utterance indices).
    # O(1) dedupe via set instead of O(n^2) scanning.
    all_indices = list(range(len(md)))
    if len(all_indices) < n_src:
        return []

    seen = set()
    pairs: List[List[int]] = []
    tries = 0
    max_tries = max(2000, target * 50)

    while len(pairs) < target and tries < max_tries:
        tries += 1
        cand = rng.sample(all_indices, n_src)
        # distinct speaker constraint
        spks = {str(md.iloc[i]["speaker_ID"]) for i in cand}
        if len(spks) != n_src:
            continue
        key = tuple(sorted(cand))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(cand)

    return pairs


def choose_noise_indices(
    utt_pairs: List[List[int]],
    speech_md: pd.DataFrame,
    noise_md: pd.DataFrame,
    rng: random.Random,
) -> List[int]:
    # Keep original spirit:
    # - Prefer non-augmented noises first.
    # - Prefer noises with length >= max speech length in mixture.
    # - Try to avoid reusing the same noise by removing selected noises.
    if "augmented" in noise_md.columns:
        pool = noise_md[noise_md["augmented"] == False].copy()  # noqa: E712
        if len(pool) < len(utt_pairs):
            pool = noise_md.copy()
    else:
        pool = noise_md.copy()

    # We'll maintain available indices and their lengths for quick filtering.
    # Using numpy arrays helps performance.
    avail_idx = pool.index.to_numpy()
    avail_len = pool["length"].to_numpy()

    noise_pairs: List[int] = []
    for pair in utt_pairs.copy():
        lengths = [int(speech_md.iloc[i]["length"]) for i in pair]
        max_len = max(lengths)

        eligible_mask = avail_len >= max_len
        eligible = avail_idx[eligible_mask]

        if eligible.size > 0:
            chosen = int(rng.choice(list(eligible)))
        else:
            # Fallback behavior close to original:
            # - for train: take the longest remaining
            # - for dev/test: drop this utterance pair (caller may resample)
            first_subset = str(speech_md.iloc[pair[0]]["subset"])
            if "train" in first_subset:
                # longest remaining
                chosen = int(avail_idx[int(np.argmax(avail_len))])
            else:
                # indicate failure by -1
                chosen = -1

        if chosen == -1:
            # dev/test: mark as missing; caller should handle
            noise_pairs.append(-1)
            continue

        noise_pairs.append(chosen)

        # remove chosen from availability
        rm = avail_idx != chosen
        avail_idx = avail_idx[rm]
        avail_len = avail_len[rm]

        if avail_idx.size == 0:
            # no more noises; allow reuse by resetting pool (best-effort)
            avail_idx = pool.index.to_numpy()
            avail_len = pool["length"].to_numpy()

    return noise_pairs


def compute_loudness_and_normalize(
    sources: List[np.ndarray],
    meter: pyln.Meter,
    rng: random.Random,
) -> Tuple[List[float], List[float], List[np.ndarray]]:
    loudness_list: List[float] = []
    target_list: List[float] = []
    normed: List[np.ndarray] = []

    for i, x in enumerate(sources):
        l0 = float(meter.integrated_loudness(x))
        loudness_list.append(l0)

        # speech target
        t = rng.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # noise target slightly lower (same as original)
        if i == len(sources) - 1:
            t = rng.uniform(MIN_LOUDNESS - 5.0, MAX_LOUDNESS - 5.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = pyln.normalize.loudness(x, l0, t)

        # If clipping happens on a single source, renormalize like original
        if float(np.max(np.abs(y))) >= 1.0:
            y = x * (MAX_AMP / (float(np.max(np.abs(x))) + EPS))
            t = float(meter.integrated_loudness(y))

        target_list.append(float(t))
        normed.append(y.astype(np.float32, copy=False))

    return loudness_list, target_list, normed


def mix_signals(signals: List[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(signals[0], dtype=np.float32)
    for s in signals:
        out += s
    return out


def renormalize_if_clipped(mixture: np.ndarray, signals: List[np.ndarray], meter: pyln.Meter) -> Tuple[List[float], bool]:
    peak = float(np.max(np.abs(mixture)))
    if peak > MAX_AMP:
        weight = MAX_AMP / (peak + EPS)
        clipped = True
    else:
        weight = 1.0
        clipped = False

    renorm_loudness = [float(meter.integrated_loudness(s * weight)) for s in signals]
    return renorm_loudness, clipped


def gain_from_loudness(original: List[float], after: List[float]) -> List[float]:
    gains: List[float] = []
    for l0, l1 in zip(original, after):
        delta = float(l1 - l0)
        gains.append(float(np.power(10.0, delta / 20.0)))
    return gains


def build_rows(
    speech_md: pd.DataFrame,
    noise_md: pd.DataFrame,
    librispeech_dir: Path,
    wham_dir: Path,
    utt_pair: List[int],
    noise_idx: int,
    n_src: int,
    meter: pyln.Meter,
    rng: random.Random,
    strict: bool,
) -> Tuple[Dict[str, object], bool]:
    """
    Build one mixture row for the mix recipe CSV.

    Returns:
        md_row: A dict containing mixture_ID, source_i_path/source_i_gain, noise_path/noise_gain.
        clipped: Whether the mixture required global renormalization due to peak clipping risk.
    """
    sources = [speech_md.iloc[i] for i in utt_pair]
    lengths = [int(s["length"]) for s in sources]
    max_len = max(lengths)

    # mixture_ID: join utterance basenames without extension, consistent across runs
    ids = [Path(str(s["origin_path"])).name.replace(".flac", "") for s in sources]
    mixture_id = "_".join(ids)

    # Load and pad/trim speech
    speech_wavs: List[np.ndarray] = []
    for s in sources:
        rel = str(s["origin_path"])
        abs_path = librispeech_dir / rel
        validate_file_exists(abs_path, strict, "LibriSpeech audio")
        wav = load_audio_mono(abs_path, strict=strict)
        speech_wavs.append(pad_or_trim(wav, max_len))

    # Load and pad/trim noise
    nrow = noise_md.loc[noise_idx]
    noise_rel = str(nrow["origin_path"])
    noise_abs = wham_dir / noise_rel
    validate_file_exists(noise_abs, strict, "WHAM noise audio")
    noise_wav = load_audio_mono(noise_abs, strict=strict)
    noise_wav = pad_or_trim(noise_wav, max_len)

    all_sources = speech_wavs + [noise_wav]

    # Loudness normalization (speech + noise)
    loud0, _targets, normed = compute_loudness_and_normalize(all_sources, meter, rng)

    # Mix and global renormalization if needed
    mixture = mix_signals(normed)
    loud1, clipped = renormalize_if_clipped(mixture, normed, meter)

    gains = gain_from_loudness(loud0, loud1)

    md_row: Dict[str, object] = {"mixture_ID": mixture_id}
    for i in range(n_src):
        md_row[f"source_{i+1}_path"] = str(sources[i]["origin_path"])
        md_row[f"source_{i+1}_gain"] = float(gains[i])

    md_row["noise_path"] = noise_rel
    md_row["noise_gain"] = float(gains[-1])

    return md_row, clipped


def create_mix_metadata_for_one_split(
    speech_md: pd.DataFrame,
    noise_md: pd.DataFrame,
    split: str,
    librispeech_dir: Path,
    wham_dir: Path,
    n_src: int,
    target_devtest: int,
    meter: pyln.Meter,
    rng: random.Random,
    strict: bool,
) -> pd.DataFrame:
    first_subset = str(speech_md.iloc[0]["subset"])
    is_train = "train" in first_subset

    if is_train:
        utt_pairs = choose_utterance_pairs_train(speech_md, n_src, rng)
    else:
        utt_pairs = choose_utterance_pairs_devtest(speech_md, n_src, target_devtest, rng)

    noise_pairs = choose_noise_indices(utt_pairs, speech_md, noise_md, rng)

    md_rows: List[Dict[str, object]] = []
    clipped_count = 0

    total = len(utt_pairs)
    for pair, nz in tqdm(list(zip(utt_pairs, noise_pairs)), total=total, desc=f"mix {split}"):
        if nz == -1:
            continue

        md_row, clipped = build_rows(
            speech_md=speech_md,
            noise_md=noise_md,
            librispeech_dir=librispeech_dir,
            wham_dir=wham_dir,
            utt_pair=pair,
            noise_idx=nz,
            n_src=n_src,
            meter=meter,
            rng=rng,
            strict=strict,
        )
        md_rows.append(md_row)
        clipped_count += int(clipped)

    print(f"[INFO] split={split} mixtures={len(md_rows)} clipped={clipped_count}")
    return pd.DataFrame(md_rows)


def main() -> None:
    args = build_parser().parse_args()

    cfg = Config(
        librispeech_dir=Path(args.librispeech_dir).expanduser().resolve(),
        librispeech_md_dir=Path(args.librispeech_md_dir).expanduser().resolve(),
        wham_dir=Path(args.wham_dir).expanduser().resolve(),
        wham_md_dir=Path(args.wham_md_dir).expanduser().resolve(),
        metadata_outdir=resolve_outdir(Path(args.librispeech_dir).expanduser().resolve(), args.metadata_outdir),
        n_src=int(args.n_src),
        seed=int(args.seed),
        dev_test_mixtures=int(args.dev_test_mixtures),
        round_to=int(args.round_to),
        overwrite=bool(args.overwrite),
        strict=bool(args.strict),
    )

    ensure_outdir(cfg)

    rng = random.Random(cfg.seed)
    meter = pyln.Meter(RATE)

    dataset = f"mix{cfg.n_src}"

    librispeech_md_files = sorted([f for f in os.listdir(cfg.librispeech_md_dir) if f.endswith(".csv")])
    wham_md_files = sorted([f for f in os.listdir(cfg.wham_md_dir) if f.endswith(".csv")])

    if not librispeech_md_files:
        raise FileNotFoundError(f"No LibriSpeech metadata CSVs found in {cfg.librispeech_md_dir}")

    for speech_csv in librispeech_md_files:
        split = split_from_librispeech_filename(speech_csv)
        try:
            noise_csv = find_wham_file(wham_md_files, split)
        except FileNotFoundError as e:
            if cfg.strict:
                raise
            print(f"[WARN] {e}. Skipping {speech_csv}.")
            continue

        speech_path = cfg.librispeech_md_dir / speech_csv
        noise_path = cfg.wham_md_dir / noise_csv

        speech_md = safe_read_csv(speech_path)
        noise_md = safe_read_csv(noise_path)

        out_recipe = cfg.metadata_outdir / f"{dataset}_{speech_csv}"

        if out_recipe.exists() and not cfg.overwrite:
            print(f"[SKIP] {out_recipe} exists (use --overwrite to regenerate)")
            continue

        print(f"[INFO] split={split} | LibriSpeech={speech_csv} | WHAM={noise_csv}")
        md_df = create_mix_metadata_for_one_split(
            speech_md=speech_md,
            noise_md=noise_md,
            split=split,
            librispeech_dir=cfg.librispeech_dir,
            wham_dir=cfg.wham_dir,
            n_src=cfg.n_src,
            target_devtest=cfg.dev_test_mixtures,
            meter=meter,
            rng=rng,
            strict=cfg.strict,
        )

        md_df = maybe_round(md_df, cfg.round_to)

        md_df.to_csv(out_recipe, index=False)
        print(f"[OK] wrote {out_recipe} (rows={len(md_df)})")



if __name__ == "__main__":
    main()
