#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create WHAM noise metadata CSVs (train/dev/test).

This script scans a WHAM noise root directory (typically containing subfolders: tr/cv/tt)
and writes exactly three CSV files under ./metadata/ (relative to the current working directory):

  - ./metadata/train.csv  (from tr/)
  - ./metadata/dev.csv    (from cv/)
  - ./metadata/test.csv   (from tt/)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import soundfile as sf
from tqdm import tqdm

DEFAULT_MIN_SECONDS = 3.0
DEFAULT_EXPECTED_SR = 16000
DEFAULT_EXPECTED_CHANNELS = 2

# Convention: augmented noise produced by speed perturbation uses tokens like "sp08"/"sp12".
# Use token boundaries to avoid accidental substring matches.
AUGMENT_RE = re.compile(r"(?:^|[_\-])sp(?:08|12)(?:[_\-\.]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class ScanConfig:
    """Configuration controlling scanning, validation, and filtering behavior."""
    wham_dir: str
    min_seconds: float
    expected_sr: int
    expected_channels: int
    strict: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic WHAM noise metadata CSVs (train/dev/test) under ./metadata/."
    )
    parser.add_argument(
        "--wham_dir",
        type=str,
        required=True,
        help="Path to WHAM noise root directory (should contain tr/cv/tt).",
    )
    parser.add_argument(
        "--min_seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help="Filter out noise files shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--expected_sr",
        type=int,
        default=DEFAULT_EXPECTED_SR,
        help="Expected sampling rate (Hz). Set <=0 to disable SR validation.",
    )
    parser.add_argument(
        "--expected_channels",
        type=int,
        default=DEFAULT_EXPECTED_CHANNELS,
        help="Expected channel count. Set <=0 to disable channel validation.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, validation/read errors become fatal instead of being skipped.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = ScanConfig(
        wham_dir=args.wham_dir,
        min_seconds=args.min_seconds,
        expected_sr=args.expected_sr,
        expected_channels=args.expected_channels,
        strict=args.strict,
    )

    wham_dir = Path(cfg.wham_dir).expanduser().resolve()
    if not wham_dir.exists():
        raise FileNotFoundError(f"--wham_dir does not exist: {wham_dir}")

    # Output directory is ./metadata relative to the current working directory.
    metadata_dir = Path.cwd() / "metadata/wham"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    subset_map: Dict[str, str] = {
        "tr": "train",
        "cv": "dev",
        "tt": "test",
    }

    # Fixed processing order to guarantee determinism.
    for subset_token in ("tr", "cv", "tt"):
        out_name = subset_map[subset_token]
        out_csv = metadata_dir / f"{out_name}.csv"

        df = scan_subset(wham_dir=wham_dir, subset_token=subset_token, cfg=cfg)
        df = apply_min_duration_filter(df, cfg)

        # Deterministic ordering: primarily by length, then by path.
        df = df.sort_values(["length", "origin_path"], ascending=[True, True]).reset_index(drop=True)

        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} (rows={len(df)})")


def scan_subset(wham_dir: Path, subset_token: str, cfg: ScanConfig) -> pd.DataFrame:
    """
    Scan one subset directory (tr/cv/tt) and return a DataFrame with a stable schema.

    Schema matches the original script contract:
      - length (samples)
      - origin_path (absolute path)
      - augmented (bool)

    Extra columns are intentionally avoided to keep compatibility and the output contract tight.
    """
    subset_dir = wham_dir / subset_token
    if not subset_dir.exists():
        msg = f"Missing subset directory: {subset_dir}"
        if cfg.strict:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg} (skipping)")
        return pd.DataFrame(columns=["length", "origin_path", "augmented"])

    # rglob order is filesystem-dependent; explicit sorting ensures determinism.
    wav_paths = sorted(subset_dir.rglob("*.wav"))

    rows: List[dict] = []
    for p in tqdm(wav_paths, desc=f"scan {subset_token}", total=len(wav_paths)):
        try:
            info = sf.info(str(p))
        except Exception as e:
            msg = f"Failed to read audio info: {p} ({e})"
            if cfg.strict:
                raise RuntimeError(msg) from e
            print(f"[WARN] {msg} (skipping)")
            continue

        # Optional validation (disable by setting <=0).
        if cfg.expected_sr > 0 and info.samplerate != cfg.expected_sr:
            msg = f"Unexpected SR={info.samplerate} (expected {cfg.expected_sr}): {p}"
            if cfg.strict:
                raise ValueError(msg)
            print(f"[WARN] {msg} (skipping)")
            continue

        if cfg.expected_channels > 0 and info.channels != cfg.expected_channels:
            msg = f"Unexpected channels={info.channels} (expected {cfg.expected_channels}): {p}"
            if cfg.strict:
                raise ValueError(msg)
            print(f"[WARN] {msg} (skipping)")
            continue

        # "augmented" is defined by the speed-perturbation token convention (sp08/sp12).
        augmented = bool(AUGMENT_RE.search(p.stem)) or bool(AUGMENT_RE.search(p.as_posix()))

        rows.append(
            {
                "length": int(info.frames),
                "origin_path": str(p),
                "augmented": bool(augmented),
            }
        )

    return pd.DataFrame(rows, columns=["length", "origin_path", "augmented"])


def apply_min_duration_filter(df: pd.DataFrame, cfg: ScanConfig) -> pd.DataFrame:
    """
    Apply minimum-duration filtering.

    To preserve the original 3-column schema, we filter using a sample threshold.
    If expected_sr is disabled, we fall back to DEFAULT_EXPECTED_SR for the threshold.

    Recommendation for strict reproducibility: keep --expected_sr enabled.
    """
    if df.empty:
        return df

    sr_for_threshold = cfg.expected_sr if cfg.expected_sr > 0 else DEFAULT_EXPECTED_SR
    min_samples = int(round(cfg.min_seconds * sr_for_threshold))
    return df[df["length"] >= min_samples]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        raise

