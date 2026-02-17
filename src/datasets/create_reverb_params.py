#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplerConfig:
    mic_width: float = 0.05
    source_num: int = 2
    min_spk_spk_theta_dist: float = 15.0
    reverb_level: str = "medium"  # {"low","medium","high"}
    num_mics: int = 6


@dataclass(frozen=True)
class RoomRanges:
    room_x: Tuple[float, float] = (5.0, 10.0)
    room_y: Tuple[float, float] = (5.0, 10.0)
    room_z: Tuple[float, float] = (3.0, 4.0)

    array_xy_jitter: Tuple[float, float] = (-0.2, 0.2)
    array_z: Tuple[float, float] = (1.0, 2.0)

    doa_deg: Tuple[int, int] = (0, 180)  # randint low..high-1, matches your original
    src_dist: Tuple[float, float] = (0.75, 2.0)

    t60_low: Tuple[float, float] = (0.1, 0.3)
    t60_medium: Tuple[float, float] = (0.2, 0.6)
    t60_high: Tuple[float, float] = (0.4, 1.0)


def _check_min_angular_distance(doas: List[float], min_dist_deg: float) -> bool:
    for i in range(len(doas)):
        for j in range(i + 1, len(doas)):
            if abs(doas[i] - doas[j]) < min_dist_deg:
                return False
    return True


def draw_params(
    rng: np.random.Generator,
    cfg: SamplerConfig,
    ranges: RoomRanges = RoomRanges(),
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[int], float, List[float]]:
    """
    Sample one room configuration.

    Returns:
        room_dim: (3,)
        mics: (num_mics, 3)
        spk_pos: list of (3,)
        spk_doa: list of int (deg)
        t60: float
        spk_dist: list of float
    """
    # room (same distribution as original) :contentReference[oaicite:2]{index=2}
    room_dim = np.array(
        [
            rng.uniform(*ranges.room_x),
            rng.uniform(*ranges.room_y),
            rng.uniform(*ranges.room_z),
        ],
        dtype=np.float32,
    )

    # microphone array center (same logic as original) :contentReference[oaicite:3]{index=3}
    array_x = room_dim[0] / 2 + rng.uniform(*ranges.array_xy_jitter)
    array_y = room_dim[1] / 2 + rng.uniform(*ranges.array_xy_jitter)
    array_z = rng.uniform(*ranges.array_z)

    # linear array along x (same offsets as original) :contentReference[oaicite:4]{index=4}
    offsets = (np.arange(cfg.num_mics) - (cfg.num_mics - 1) / 2.0) * cfg.mic_width
    mics = np.stack([np.array([array_x + off, array_y, array_z], dtype=np.float32) for off in offsets], axis=0)

    # speakers (keep rejection sampling + min DOA distance as original) :contentReference[oaicite:5]{index=5}
    while True:
        spk_doa: List[int] = []
        spk_dist: List[float] = []
        spk_pos: List[np.ndarray] = []

        for _ in range(cfg.source_num):
            theta = int(rng.integers(ranges.doa_deg[0], ranges.doa_deg[1]))  # [0,180)
            dist = float(rng.uniform(*ranges.src_dist))

            spk_doa.append(theta)
            spk_dist.append(dist)

            rad = theta / 180.0 * np.pi
            pos = np.array(
                [
                    array_x + dist * np.cos(rad),
                    array_y + dist * np.sin(rad),
                    array_z,
                ],
                dtype=np.float32,
            )
            spk_pos.append(pos)

        if _check_min_angular_distance(spk_doa, cfg.min_spk_spk_theta_dist):
            break

    # T60 (same bins as original) :contentReference[oaicite:6]{index=6}
    if cfg.reverb_level == "high":
        t60 = float(rng.uniform(*ranges.t60_high))
    elif cfg.reverb_level == "medium":
        t60 = float(rng.uniform(*ranges.t60_medium))
    elif cfg.reverb_level == "low":
        t60 = float(rng.uniform(*ranges.t60_low))
    else:
        raise ValueError(f"Unknown reverb_level='{cfg.reverb_level}'. Expected: low|medium|high")

    return room_dim, mics, spk_pos, spk_doa, t60, spk_dist


def _wide_schema_row(
    mixture_id: str,
    room_dim: np.ndarray,
    mics: np.ndarray,
    spk_pos: List[np.ndarray],
    spk_doa: List[int],
    t60: float,
) -> Dict[str, object]:
    """
    Default output schema matches your current Libri2Mix driver:
      room_x room_y room_z
      mic1_x..mic6_x mic_y mic_z
      s1_x s1_y s1_z ... s2_x s2_y s2_z
      s1_doa s2_doa
      T60
    """
    num_mics = mics.shape[0]
    num_src = len(spk_pos)

    row: Dict[str, object] = {
        "mixture_ID": mixture_id,
        "room_x": float(room_dim[0]),
        "room_y": float(room_dim[1]),
        "room_z": float(room_dim[2]),
        "mic_y": float(mics[0, 1]),
        "mic_z": float(mics[0, 2]),
        "T60": float(t60),
    }

    # mic*_x
    for j in range(num_mics):
        row[f"mic{j+1}_x"] = float(mics[j, 0])

    # s*_x/y/z + s*_doa
    for i in range(num_src):
        row[f"s{i+1}_x"] = float(spk_pos[i][0])
        row[f"s{i+1}_y"] = float(spk_pos[i][1])
        row[f"s{i+1}_z"] = float(spk_pos[i][2])
        row[f"s{i+1}_doa"] = int(spk_doa[i])

    return row


def _infer_output_columns(num_mics: int, num_src: int) -> List[str]:
    cols = ["mixture_ID", "room_x", "room_y", "room_z"]
    cols += [f"mic{j+1}_x" for j in range(num_mics)]
    cols += ["mic_y", "mic_z"]
    for i in range(num_src):
        cols += [f"s{i+1}_x", f"s{i+1}_y", f"s{i+1}_z"]
    for i in range(num_src):
        cols += [f"s{i+1}_doa"]
    cols += ["T60"]
    return cols


def _read_mixture_ids(metadata_csv: Path, id_column: str = "mixture_ID") -> List[str]:
    df = pd.read_csv(metadata_csv)
    if id_column not in df.columns:
        raise KeyError(f"Column '{id_column}' not found in {metadata_csv}. Available columns: {list(df.columns)}")
    return df[id_column].astype(str).tolist()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample room/mic/source parameters for each mixture_ID and write reverb-params CSV(s)."
    )

    # Batch mode (recommended): generate params for all mix2_*.csv in a directory
    p.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="If set, generate params for all mix2_*.csv under this directory.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory used with --metadata_dir (same filenames as inputs).",
    )

    # Single-file mode (legacy defaults preserved)
    p.add_argument(
        "--metadata_csv",
        type=str,
        default=str(Path("metadata") / "Libri2Mix" / "libri2mix_train-clean-360.csv"),
        help="Input mixture metadata CSV containing mixture_ID (single-file mode).",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        default=str(Path("reverb_params") / "Libri2Mix" / "libri2mix_train-clean-360.csv"),
        help="Output reverb-params CSV path (single-file mode).",
    )

    p.add_argument("--id_column", type=str, default="mixture_ID")

    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--num_mics", type=int, default=6)
    p.add_argument("--n_src", type=int, default=2)
    p.add_argument("--mic_width", type=float, default=0.05)
    p.add_argument("--min_spk_spk_theta_dist", type=float, default=15.0)
    p.add_argument("--reverb_level", type=str, default="medium", choices=["low", "medium", "high"])

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--progress", action="store_true", help="Show tqdm progress bar.")

    return p



def main() -> None:
    args = build_arg_parser().parse_args()

    sampler_cfg = SamplerConfig(
        mic_width=float(args.mic_width),
        source_num=int(args.n_src),
        min_spk_spk_theta_dist=float(args.min_spk_spk_theta_dist),
        reverb_level=str(args.reverb_level),
        num_mics=int(args.num_mics),
    )
    cols = _infer_output_columns(num_mics=sampler_cfg.num_mics, num_src=sampler_cfg.source_num)

    # Helper to run one file
    def run_one(metadata_csv: Path, out_csv: Path) -> None:
        if not metadata_csv.exists():
            raise FileNotFoundError(f"metadata csv not found: {metadata_csv}")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {out_csv}. Use --overwrite to overwrite.")

        mixture_ids = _read_mixture_ids(metadata_csv, id_column=args.id_column)

        rng = np.random.default_rng(int(args.seed))
        rows: List[Dict[str, object]] = []

        iterator: Iterable[str]
        if args.progress:
            from tqdm import tqdm
            iterator = tqdm(mixture_ids, desc=f"sample params ({metadata_csv.name})", total=len(mixture_ids))
        else:
            iterator = mixture_ids

        for mid in iterator:
            room_dim, mics, spk_pos, spk_doa, t60, _spk_dist = draw_params(rng=rng, cfg=sampler_cfg)
            rows.append(_wide_schema_row(mid, room_dim, mics, spk_pos, spk_doa, t60))

        df = pd.DataFrame.from_records(rows, columns=cols)
        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} (rows={len(df)})")

    # Batch mode
    if args.metadata_dir is not None:
        md_dir = Path(args.metadata_dir).expanduser().resolve()
        if not md_dir.exists():
            raise FileNotFoundError(f"--metadata_dir not found: {md_dir}")

        if args.out_dir is None:
            raise ValueError("When using --metadata_dir, you must also provide --out_dir.")
        out_dir = Path(args.out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pattern is hard-coded as requested
        md_files = sorted(md_dir.glob("mix2_*.csv"))
        if not md_files:
            raise FileNotFoundError(f"No files matched pattern 'mix2_*.csv' under: {md_dir}")

        for md_csv in md_files:
            out_csv = out_dir / md_csv.name
            run_one(md_csv, out_csv)

        return

    # Single-file mode (legacy)
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    run_one(metadata_csv, out_csv)



if __name__ == "__main__":
    main()
