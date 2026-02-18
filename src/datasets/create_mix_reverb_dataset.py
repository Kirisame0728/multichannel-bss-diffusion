#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
from tqdm import tqdm


EPS = 1e-10
SRC_RATE = 16000  # LibriSpeech original sampling rate (Hz)


@dataclass(frozen=True)
class Config:
    librispeech_dir: Path
    wham_dir: Path
    metadata_csv: Path
    params_csv: Path
    out_dir: Path

    n_src: int = 2
    fs: int = 8000
    mode: str = "min"  # {"min","max"}
    split_dir: str = "test_eval92"

    early_ms: float = 50.0
    add_noise: bool = False
    anechoic: bool = False
    noise_atten_db: float = 12.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create multi-channel reverberant mixtures with SMS-WSJ-like early/tail/observation structure."
    )
    p.add_argument("--librispeech_dir", type=str, required=True, help="Path to LibriSpeech root directory")
    p.add_argument("--wham_dir", type=str, required=True, help="Path to WHAM noise root directory")
    p.add_argument("--metadata_csv", type=str, required=True, help="Path to ONE mix metadata CSV (recipe)")
    p.add_argument("--params_csv", type=str, required=True, help="Path to matched reverb params CSV (same mixture_IDs)")
    p.add_argument("--out_dir", type=str, required=True, help="Output dataset root directory")

    p.add_argument("--n_src", type=int, default=2, help="Number of sources (speakers) in mixtures (default: 2)")
    p.add_argument("--fs", type=int, default=8000, help="Target sampling rate for output audio (default: 8000)")
    p.add_argument("--mode", type=str, default="min", choices=["min", "max"], help="Length mode when mixing sources")
    p.add_argument(
        "--split_dir",
        type=str,
        default="test_eval92",
        help="Subfolder name under early/tail/observation (default: test_eval92)",
    )

    p.add_argument(
        "--early_ms",
        type=float,
        default=50.0,
        help="Early window length (ms) after direct-path onset (default: 50ms)",
    )
    p.add_argument(
        "--add_noise",
        action="store_true",
        help="If set, add WHAM noise to observation (same noise replicated to all mics).",
    )
    p.add_argument(
        "--anechoic",
        action="store_true",
        help="If set, force max_order=0 (anechoic). Tail will be ~0.",
    )
    p.add_argument(
        "--noise_atten_db",
        type=float,
        default=12.0,
        help="Attenuate added noise by this many dB (positive => less noise, higher SNR).",
    )
    return p


def parse_args_to_cfg() -> Config:
    args = build_parser().parse_args()
    return Config(
        librispeech_dir=Path(args.librispeech_dir).expanduser().resolve(),
        wham_dir=Path(args.wham_dir).expanduser().resolve(),
        metadata_csv=Path(args.metadata_csv).expanduser().resolve(),
        params_csv=Path(args.params_csv).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        n_src=int(args.n_src),
        fs=int(args.fs),
        mode=str(args.mode),
        split_dir=str(args.split_dir),
        early_ms=float(args.early_ms),
        add_noise=bool(args.add_noise),
        anechoic=bool(args.anechoic),
        noise_atten_db=float(args.noise_atten_db),
    )


def ensure_out_dirs(out_root: Path, split_dir: str) -> Tuple[Path, Path, Path]:
    obs_dir = out_root / "observation" / split_dir
    early_dir = out_root / "early" / split_dir
    tail_dir = out_root / "tail" / split_dir
    obs_dir.mkdir(parents=True, exist_ok=True)
    early_dir.mkdir(parents=True, exist_ok=True)
    tail_dir.mkdir(parents=True, exist_ok=True)
    return obs_dir, early_dir, tail_dir


def require_columns(df: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {where}: {missing}. Available={list(df.columns)}")


def get_list_from_recipe_row(row: pd.Series, base: str, n_src: int) -> List[object]:
    """
    base = 'source_path' -> read source_1_path, source_2_path, ...
    base = 'source_gain' -> read source_1_gain, ...
    """
    out: List[object] = []
    parts = base.split("_")
    for i in range(n_src):
        parts_i = parts.copy()
        parts_i.insert(1, str(i + 1))
        col = "_".join(parts_i)
        out.append(row[col])
    return out


def extend_noise_to_length(noise: np.ndarray, target_len: int) -> np.ndarray:
    """
    Repeat WHAM noise to reach target_len using Hann cross-fade (keeps original behavior).
    """
    noise = np.asarray(noise, dtype=np.float32).reshape(-1)
    if len(noise) >= target_len:
        return noise[:target_len]

    noise_ex = noise
    window = np.hanning(SRC_RATE + 1).astype(np.float32)
    i_w = window[: len(window) // 2 + 1]
    d_w = window[len(window) // 2 :: -1]

    while len(noise_ex) < target_len:
        # cross-fade tail of current noise_ex with start of original noise
        noise_ex = np.concatenate(
            (
                noise_ex[: len(noise_ex) - len(d_w)],
                noise_ex[len(noise_ex) - len(d_w) :] * d_w + noise[: len(i_w)] * i_w,
                noise[len(i_w) :],
            ),
            axis=0,
        )
    return noise_ex[:target_len]


def read_sources_from_recipe(
    row: pd.Series,
    cfg: Config,
) -> Tuple[str, List[float], List[np.ndarray]]:
    """
    Read n_src speech sources and optional mono noise.
    Returns:
        mixture_id, gain_list, sources_list
        where sources_list = [s1, s2, ..., (noise if add_noise)]
    """
    mixture_id = str(row["mixture_ID"])

    # Required recipe columns
    require_columns(
        row.to_frame().T,
        ["mixture_ID"]
        + [f"source_{i+1}_path" for i in range(cfg.n_src)]
        + [f"source_{i+1}_gain" for i in range(cfg.n_src)],
        where="metadata_csv row",
    )
    if cfg.add_noise:
        require_columns(row.to_frame().T, ["noise_path", "noise_gain"], where="metadata_csv row (noise)")

    sources_path_list = [str(x) for x in get_list_from_recipe_row(row, "source_path", cfg.n_src)]
    gains = [float(x) for x in get_list_from_recipe_row(row, "source_gain", cfg.n_src)]

    sources: List[np.ndarray] = []
    max_len = 0

    # Speech sources
    for rel in sources_path_list:
        abs_path = cfg.librispeech_dir / rel
        if not abs_path.exists():
            raise FileNotFoundError(f"Missing speech file: {abs_path} (from origin_path='{rel}')")
        s, _ = sf.read(str(abs_path), dtype="float32")
        if s.ndim > 1:
            s = s[:, 0]
        max_len = max(max_len, len(s))
        sources.append(np.asarray(s, dtype=np.float32))

    # Noise (optional)
    if cfg.add_noise:
        noise_rel = str(row["noise_path"])
        noise_abs = cfg.wham_dir / noise_rel
        if not noise_abs.exists():
            raise FileNotFoundError(f"Missing noise file: {noise_abs} (from noise_path='{noise_rel}')")
        noise, _ = sf.read(str(noise_abs), dtype="float32", stop=max_len)
        if noise.ndim > 1:
            noise = noise[:, 0]
        noise = np.asarray(noise, dtype=np.float32)
        if len(noise) < max_len:
            noise = extend_noise_to_length(noise, max_len)

        # apply metadata noise_gain, then attenuate by noise_atten_db
        noise_gain = float(row["noise_gain"]) * (10.0 ** (-cfg.noise_atten_db / 20.0))
        sources.append(noise)
        gains.append(float(noise_gain))

    return mixture_id, gains, sources


def apply_gains_and_resample(
    sources: List[np.ndarray], gains: List[float], fs_out: int
) -> List[np.ndarray]:
    """
    Apply gains then resample from SRC_RATE -> fs_out.
    """
    if len(sources) != len(gains):
        raise ValueError(f"len(sources) != len(gains): {len(sources)} vs {len(gains)}")

    out: List[np.ndarray] = []
    for s, g in zip(sources, gains):
        s = np.asarray(s, dtype=np.float32) * float(g)
        if fs_out != SRC_RATE:
            s = resample_poly(s, fs_out, SRC_RATE).astype(np.float32, copy=False)
        out.append(s)
    return out


def fit_lengths(sources: List[np.ndarray], mode: str) -> List[np.ndarray]:
    if not sources:
        return sources
    if mode == "min":
        target = min(len(s) for s in sources)
        return [s[:target] for s in sources]
    target = max(len(s) for s in sources)
    return [np.pad(s, (0, target - len(s)), mode="constant").astype(np.float32) for s in sources]


def split_rir_early_late(h: np.ndarray, fs: int, early_ms: float = 50.0, thr: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split RIR into early and late parts.

    Strategy (same spirit as original):
      - Find peak index n_peak
      - Find the first index n0 in [0, n_peak] where |h[n]| >= thr * |h[n_peak]|
      - early = h[n0 : n0+Le), late = h - early
    """
    h = np.asarray(h, dtype=np.float32).reshape(-1)
    if h.size == 0:
        return h, h

    Le = int(round(fs * early_ms / 1000.0))
    n_peak = int(np.argmax(np.abs(h)))
    peak = float(np.abs(h[n_peak]) + 1e-12)
    threshold = thr * peak

    n0 = 0
    for n in range(n_peak + 1):
        if float(np.abs(h[n])) >= threshold:
            n0 = n
            break

    h_early = np.zeros_like(h)
    end = min(len(h), n0 + Le)
    h_early[n0:end] = h[n0:end]
    h_late = h - h_early
    return h_early, h_late


def convolve_rir_multi(sig: np.ndarray, rirs: Sequence[np.ndarray]) -> np.ndarray:
    """
    sig: (T,)
    rirs: list length M, each 1D
    return: (T, M)  (trim to T)
    """
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    T = sig.shape[0]
    M = len(rirs)
    out = np.zeros((T, M), dtype=np.float32)
    for m in range(M):
        y = fftconvolve(sig, np.asarray(rirs[m], dtype=np.float32).reshape(-1)).astype(np.float32, copy=False)
        out[:, m] = y[:T]
    return out


def generate_images_full_early_late(
    room_dim: Sequence[float],
    mics_pos: Sequence[Sequence[float]],
    sources_pos: Sequence[Sequence[float]],
    t60: float,
    fs: int,
    signals: Sequence[np.ndarray],
    early_ms: float,
    anechoic: bool,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns:
      full_list  : list length n_src, each (T, M)
      early_list : list length n_src, each (T, M)
      late_list  : list length n_src, each (T, M)
    """
    n_src = len(signals)
    M = len(mics_pos)
    if len(sources_pos) != n_src:
        raise ValueError(f"len(sources_pos) != n_src: {len(sources_pos)} vs {n_src}")

    e_absorption, max_order = pra.inverse_sabine(float(t60), room_dim)
    if anechoic:
        max_order = 0

    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    for k in range(n_src):
        room.add_source(sources_pos[k], signal=None)

    room.add_microphone_array(pra.MicrophoneArray(np.array(mics_pos, dtype=np.float32).T, fs))
    room.compute_rir()  # room.rir[m][k]

    T = int(signals[0].shape[0])

    full_list: List[np.ndarray] = []
    early_list: List[np.ndarray] = []
    late_list: List[np.ndarray] = []

    for k in range(n_src):
        rirs_full = [room.rir[m][k] for m in range(M)]
        rirs_early: List[np.ndarray] = []
        rirs_late: List[np.ndarray] = []
        for m in range(M):
            h_e, h_l = split_rir_early_late(rirs_full[m], fs=fs, early_ms=early_ms)
            rirs_early.append(h_e)
            rirs_late.append(h_l)

        x_full = convolve_rir_multi(signals[k], rirs_full)[:T, :]
        x_early = convolve_rir_multi(signals[k], rirs_early)[:T, :]
        x_late = convolve_rir_multi(signals[k], rirs_late)[:T, :]

        full_list.append(x_full)
        early_list.append(x_early)
        late_list.append(x_late)

    return full_list, early_list, late_list


def sum_multichannel(images: Sequence[np.ndarray]) -> np.ndarray:
    mix = np.zeros_like(images[0], dtype=np.float32)
    for x in images:
        mix += np.asarray(x, dtype=np.float32)
    return mix


def params_row_to_geometry(pr: pd.Series, n_src: int) -> Tuple[List[float], List[List[float]], List[List[float]], float]:
    """
    Extract room_dim, mics_pos, sources_pos, T60 from params row.
    Supports variable n_mics based on available mic{j}_x columns.
    """
    require_columns(pr.to_frame().T, ["room_x", "room_y", "room_z", "mic_y", "mic_z", "T60"], where="params_csv row")

    room_dim = [float(pr["room_x"]), float(pr["room_y"]), float(pr["room_z"])]
    mic_y = float(pr["mic_y"])
    mic_z = float(pr["mic_z"])

    # Determine number of mics by scanning mic{j}_x columns
    mic_x_cols = sorted([c for c in pr.index if c.startswith("mic") and c.endswith("_x")], key=lambda s: int(s[3:-2]))
    if not mic_x_cols:
        raise KeyError("No mic*_x columns found in params_csv row.")
    mics_pos = [[float(pr[c]), mic_y, mic_z] for c in mic_x_cols]

    sources_pos: List[List[float]] = []
    for i in range(n_src):
        sx, sy, sz = f"s{i+1}_x", f"s{i+1}_y", f"s{i+1}_z"
        require_columns(pr.to_frame().T, [sx, sy, sz], where="params_csv row (sources)")
        sources_pos.append([float(pr[sx]), float(pr[sy]), float(pr[sz])])

    t60 = float(pr["T60"])
    return room_dim, mics_pos, sources_pos, t60


def run(cfg: Config) -> None:
    if not cfg.metadata_csv.exists():
        raise FileNotFoundError(f"--metadata_csv not found: {cfg.metadata_csv}")
    if not cfg.params_csv.exists():
        raise FileNotFoundError(f"--params_csv not found: {cfg.params_csv}")
    if not cfg.librispeech_dir.exists():
        raise FileNotFoundError(f"--librispeech_dir not found: {cfg.librispeech_dir}")
    if not cfg.wham_dir.exists():
        # still required by CLI; only truly needed when --add_noise
        if cfg.add_noise:
            raise FileNotFoundError(f"--wham_dir not found: {cfg.wham_dir}")

    obs_dir, early_dir, tail_dir = ensure_out_dirs(cfg.out_dir, cfg.split_dir)

    recipe = pd.read_csv(cfg.metadata_csv, engine="python")
    params = pd.read_csv(cfg.params_csv, engine="python")

    require_columns(recipe, ["mixture_ID"], where="metadata_csv")
    require_columns(params, ["mixture_ID"], where="params_csv")

    params_by_id = params.set_index("mixture_ID", drop=False)

    # Faster than iterrows; still gives named attributes
    rows_iter = recipe.itertuples(index=False)

    for row in tqdm(list(rows_iter), total=len(recipe), desc=f"render {cfg.split_dir}"):
        row_s = pd.Series(row._asdict())

        mix_id, gains, sources = read_sources_from_recipe(row_s, cfg)
        sources = apply_gains_and_resample(sources, gains, cfg.fs)
        sources = fit_lengths(sources, cfg.mode)

        spk_signals = sources[: cfg.n_src]
        noise = sources[-1] if cfg.add_noise else None

        if mix_id not in params_by_id.index:
            raise KeyError(f"mixture_ID '{mix_id}' not found in params_csv: {cfg.params_csv}")
        pr = params_by_id.loc[mix_id]

        room_dim, mics_pos, sources_pos, t60 = params_row_to_geometry(pr, cfg.n_src)

        full_list, early_list, late_list = generate_images_full_early_late(
            room_dim=room_dim,
            mics_pos=mics_pos,
            sources_pos=sources_pos,
            t60=t60,
            fs=cfg.fs,
            signals=spk_signals,
            early_ms=cfg.early_ms,
            anechoic=cfg.anechoic,
        )

        # write early/tail per speaker
        for k in range(cfg.n_src):
            sf.write(str(early_dir / f"{mix_id}_{k}.wav"), early_list[k], cfg.fs)
            sf.write(str(tail_dir / f"{mix_id}_{k}.wav"), late_list[k], cfg.fs)

        # observation = sum of full images (+ optional noise replicated)
        observation = sum_multichannel(full_list)

        if cfg.add_noise:
            if noise is None:
                raise RuntimeError("Internal error: add_noise=True but noise is None.")
            T = observation.shape[0]
            noise = noise[:T]
            observation = observation + noise.reshape(-1, 1)

        sf.write(str(obs_dir / f"{mix_id}.wav"), observation, cfg.fs)


def main() -> None:
    cfg = parse_args_to_cfg()
    run(cfg)


if __name__ == "__main__":
    main()
