#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LibriSpeech Metadata Generator

This script builds utterance-level metadata (one row per .flac file) for the LibriSpeech dataset.
It scans the selected LibriSpeech subsets (e.g., train-clean-100, dev-clean, test-clean, etc.)
and writes CSV files to:

    ./metadata/librispeech/<subset>.csv

Output schema (stable):
    speaker_ID, sex, subset, length, origin_path

Design goals:
  - Determinism:
      * Fixed subset processing order.
      * Sorted file discovery.
      * Deterministic output ordering.
  - Robustness:
      * Clear errors/warnings for missing files or unexpected formats.
      * Optional strict mode to fail fast.
  - Performance:
      * Collect rows in memory and build a DataFrame once.
  - Compatibility:
      * Preserves the original column contract and filtering behavior (<3s removed).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Constants (match common LibriSpeech assumptions)
# -----------------------------
DEFAULT_RATE = 16000
DEFAULT_MIN_SECONDS = 3.0

# Canonical subset names in LibriSpeech recipes.
CANONICAL_SUBSETS = (
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
)


@dataclass(frozen=True)
class Config:
    librispeech_dir: str
    output_dir: str
    min_seconds: float
    rate: int
    strict: bool
    overwrite: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create LibriSpeech metadata CSVs.")
    parser.add_argument(
        "--librispeech_dir",
        type=str,
        required=True,
        help="Path to LibriSpeech root directory (must contain SPEAKERS.TXT and subset folders).",
    )

    # Output is fixed to ./metadata/librispeech by default per your requirement.
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.cwd() / "metadata" / "librispeech"),
        help="Output directory for CSV metadata (default: ./metadata/librispeech).",
    )

    parser.add_argument(
        "--min_seconds",
        type=float,
        default=DEFAULT_MIN_SECONDS,
        help="Filter out utterances shorter than this duration in seconds.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        help="Sample rate used to convert seconds to a sample threshold (default: 16000).",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="If set, errors (missing speaker, unreadable audio, missing files) are fatal.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, regenerate CSVs even if they already exist in the output directory.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = Config(
        librispeech_dir=args.librispeech_dir,
        output_dir=args.output_dir,
        min_seconds=args.min_seconds,
        rate=args.rate,
        strict=args.strict,
        overwrite=args.overwrite,
    )

    librispeech_dir = Path(cfg.librispeech_dir).expanduser().resolve()
    if not librispeech_dir.exists():
        raise FileNotFoundError(f"--librispeech_dir does not exist: {librispeech_dir}")

    out_dir = Path(cfg.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers_map = load_speakers_map(librispeech_dir, strict=cfg.strict)

    subsets_to_process = resolve_subsets_to_process(
        librispeech_dir=librispeech_dir,
        out_dir=out_dir,
        overwrite=cfg.overwrite,
    )

    # Deterministic processing order: use canonical ordering.
    for subset in subsets_to_process:
        df = build_subset_metadata(
            librispeech_dir=librispeech_dir,
            subset=subset,
            speakers_map=speakers_map,
            strict=cfg.strict,
        )

        # Filter out utterances shorter than min_seconds.
        min_samples = int(round(cfg.min_seconds * cfg.rate))
        df = df[df["length"] >= min_samples]

        # Deterministic ordering within a subset: sort by length then by path.
        df = df.sort_values(["length", "origin_path"], ascending=[True, True]).reset_index(drop=True)

        out_csv = out_dir / f"{subset}.csv"
        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} (rows={len(df)})")


def load_speakers_map(librispeech_dir: Path, strict: bool) -> Dict[int, Tuple[str, str]]:
    """
    Parse SPEAKERS.TXT and return a mapping:
        speaker_id -> (sex, subset)

    Notes:
      - Some LibriSpeech distributions may contain edge cases in SPEAKERS.TXT.
      - The original script appended speaker_id=60 manually due to parsing issues.
        Here we handle parsing robustly and keep the same fallback for compatibility.
    """
    speakers_txt = librispeech_dir / "SPEAKERS.TXT"
    if not speakers_txt.exists():
        msg = f"Missing SPEAKERS.TXT at: {speakers_txt}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}. Speaker sex/subset may be incomplete.", file=sys.stderr)
        return {}

    # Robust parsing:
    # SPEAKERS.TXT is pipe-separated with a header block; original code used skiprows=11.
    # We keep the same assumption but guard against format drift.
    try:
        df = pd.read_csv(
            speakers_txt,
            sep="|",
            skiprows=11,
            header=0,
            names=["speaker_ID", "sex", "subset", "minutes", "names"],
            skipinitialspace=True,
            on_bad_lines="skip",  # pandas>=1.3
            engine="python",
        )
    except TypeError:
        # Compatibility for older pandas that used error_bad_lines
        df = pd.read_csv(
            speakers_txt,
            sep="|",
            skiprows=11,
            header=0,
            names=["speaker_ID", "sex", "subset", "minutes", "names"],
            skipinitialspace=True,
            error_bad_lines=False,  # type: ignore
            engine="python",
        )

    # Normalize and keep only required fields.
    df = df.drop(columns=[c for c in ["minutes", "names"] if c in df.columns], errors="ignore")
    for col in ["sex", "subset"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Compatibility fallback: speaker 60 entry sometimes breaks delimiter-based parsing.
    # Preserve original behavior: inject a plausible mapping if missing.
    if "speaker_ID" in df.columns:
        try:
            speaker_ids = set(int(x) for x in df["speaker_ID"].dropna().tolist())
        except Exception:
            speaker_ids = set()
    else:
        speaker_ids = set()

    if 60 not in speaker_ids:
        df.loc[len(df)] = [60, "M", "train-clean-100"]

    mapping: Dict[int, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        try:
            spk = int(row["speaker_ID"])
            sex = str(row["sex"]).strip()
            subset = str(row["subset"]).strip()
            if sex and subset:
                mapping[spk] = (sex, subset)
        except Exception:
            # Skip malformed entries quietly unless strict.
            if strict:
                raise
            continue

    return mapping


def resolve_subsets_to_process(librispeech_dir: Path, out_dir: Path, overwrite: bool) -> List[str]:
    """
    Determine which LibriSpeech subset folders exist locally and should be processed.

    Determinism:
      - Return subsets in canonical order.
    """
    existing_subsets = {p.name for p in librispeech_dir.iterdir() if p.is_dir()}
    available = [s for s in CANONICAL_SUBSETS if s in existing_subsets]

    if overwrite:
        return available

    # Skip subsets that already have an output CSV.
    to_process: List[str] = []
    for subset in available:
        if not (out_dir / f"{subset}.csv").exists():
            to_process.append(subset)
    return to_process


def build_subset_metadata(
    librispeech_dir: Path,
    subset: str,
    speakers_map: Dict[int, Tuple[str, str]],
    strict: bool,
) -> pd.DataFrame:
    """
    Build utterance-level metadata for a single LibriSpeech subset.

    Output columns:
      speaker_ID: string (kept as string for backward compatibility with original script)
      sex: string
      subset: string (speaker subset from SPEAKERS.TXT, not necessarily equal to folder name)
      length: int (number of samples)
      origin_path: string (path relative to librispeech_dir)
    """
    subset_dir = librispeech_dir / subset
    if not subset_dir.exists():
        msg = f"Subset directory does not exist: {subset_dir}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg} (skipping)", file=sys.stderr)
        return pd.DataFrame(columns=["speaker_ID", "sex", "subset", "length", "origin_path"])

    # Deterministic discovery: glob + explicit sorting.
    flac_paths = sorted(
        glob.glob(str(subset_dir / "**" / "*.flac"), recursive=True)
    )

    rows: List[dict] = []
    for fp in tqdm(flac_paths, desc=f"scan {subset}", total=len(flac_paths)):
        p = Path(fp)

        # Speaker ID is encoded in filename: <speaker>-<chapter>-<utterance>.flac
        try:
            spk_id_str = p.name.split("-")[0]
            spk_id = int(spk_id_str)
        except Exception as e:
            msg = f"Failed to parse speaker ID from filename: {p}"
            if strict:
                raise ValueError(msg) from e
            print(f"[WARN] {msg} (skipping)", file=sys.stderr)
            continue

        # Read audio length in samples.
        try:
            length = int(sf.info(str(p)).frames)
        except Exception as e:
            msg = f"Failed to read audio info: {p}"
            if strict:
                raise RuntimeError(msg) from e
            print(f"[WARN] {msg} (skipping)", file=sys.stderr)
            continue

        # Lookup speaker sex/subset; if missing, keep placeholders.
        if speakers_map and spk_id in speakers_map:
            sex, spk_subset = speakers_map[spk_id]
        else:
            sex, spk_subset = ("", "")
            msg = f"Speaker {spk_id} not found in SPEAKERS.TXT mapping for file: {p}"
            if strict:
                raise KeyError(msg)
            print(f"[WARN] {msg}", file=sys.stderr)

        rel_path = os.path.relpath(str(p), str(librispeech_dir))

        rows.append(
            {
                "speaker_ID": str(spk_id_str),
                "sex": sex,
                "subset": spk_subset,
                "length": length,
                "origin_path": rel_path,
            }
        )

    return pd.DataFrame(rows, columns=["speaker_ID", "sex", "subset", "length", "origin_path"])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", file=sys.stderr)
        raise
