#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

DEFAULT_RATE = 16000
DEFAULT_MIN_SECONDS = 3.0

CANONICAL_SUBSETS = (
    "dev-clean",
    "test-clean",
    "train-clean-100",
    "train-clean-360",
)


@dataclass(frozen=True)
class Config:
    librispeech_dir: Optional[str]     # only used in --all mode
    output_dir: str
    min_seconds: float
    rate: int
    strict: bool
    overwrite: bool
    all_mode: bool
    subset_dirs: Dict[str, str]        # subset -> dir (used in explicit mode)
    speakers_txt: Optional[str]        # resolved SPEAKERS.TXT path (or None)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create LibriSpeech utterance-level metadata CSVs."
    )

    mode = p.add_mutually_exclusive_group(required=True)

    # Mode A: unified root, scan available subsets
    mode.add_argument(
        "--all",
        action="store_true",
        help="Scan subsets under --librispeech_dir (unified LibriSpeech root layout).",
    )

    # Mode B: explicit subset directories (allow any subset to be provided)
    # Each option takes a directory path.
    mode.add_argument(
        "--explicit",
        action="store_true",
        help="Use explicitly provided subset directories (e.g., --dev-clean DIR).",
    )

    p.add_argument(
        "--librispeech_dir",
        type=str,
        default=None,
        help="LibriSpeech root directory (required when using --all). In --explicit mode, can be used to locate SPEAKERS.TXT.",
    )

    # Optional explicit SPEAKERS.TXT override (useful in --explicit mode).
    p.add_argument(
        "--speakers_txt",
        type=str,
        default=None,
        help="Optional path to SPEAKERS.TXT. Overrides auto-detection.",
    )

    # Explicit subset directory arguments (only meaningful with --explicit)
    for s in CANONICAL_SUBSETS:
        p.add_argument(
            f"--{s}",
            dest=f"subset_{s}",
            type=str,
            default=None,
            help=f"Directory for subset '{s}' (used with --explicit).",
        )

    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path.cwd() / "metadata" / "librispeech"),
        help="Output directory for CSV metadata (default: ./metadata/librispeech).",
    )
    p.add_argument("--min_seconds", type=float, default=DEFAULT_MIN_SECONDS)
    p.add_argument("--rate", type=int, default=DEFAULT_RATE)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    subset_dirs: Dict[str, str] = {}
    if args.explicit:
        for s in CANONICAL_SUBSETS:
            v = getattr(args, f"subset_{s}")
            if v is not None:
                subset_dirs[s] = v

        if not subset_dirs:
            raise ValueError(
                "In --explicit mode, you must provide at least one subset directory, "
                "e.g., --dev-clean /path/to/dev-clean"
            )

        librispeech_dir = Path(args.librispeech_dir).expanduser().resolve() if args.librispeech_dir else None

    else:
        # --all mode
        if not args.librispeech_dir:
            raise ValueError("--librispeech_dir is required when using --all.")
        librispeech_dir = Path(args.librispeech_dir).expanduser().resolve()
        if not librispeech_dir.exists():
            raise FileNotFoundError(f"--librispeech_dir does not exist: {librispeech_dir}")

        subset_dirs = resolve_subsets_under_root(librispeech_dir)

    # Resolve SPEAKERS.TXT robustly:
    # 1) --speakers_txt (explicit)
    # 2) <librispeech_dir>/SPEAKERS.TXT (if provided)
    # 3) walk up from each explicit subset directory and search for SPEAKERS.TXT
    speakers_txt_path = resolve_speakers_txt(
        speakers_txt_arg=args.speakers_txt,
        librispeech_dir=librispeech_dir,
        subset_dirs=[Path(v) for v in subset_dirs.values()],
        max_up_levels=6,
    )

    cfg = Config(
        librispeech_dir=str(librispeech_dir) if librispeech_dir else None,
        output_dir=str(Path(args.output_dir).expanduser().resolve()),
        min_seconds=args.min_seconds,
        rate=args.rate,
        strict=args.strict,
        overwrite=args.overwrite,
        all_mode=bool(args.all),
        subset_dirs=subset_dirs,
        speakers_txt=str(speakers_txt_path) if speakers_txt_path else None,
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speakers_map = load_speakers_map(Path(cfg.speakers_txt) if cfg.speakers_txt else None, strict=cfg.strict)

    if cfg.speakers_txt:
        print(f"[INFO] Using SPEAKERS.TXT: {cfg.speakers_txt}")
    else:
        print("[INFO] SPEAKERS.TXT not found; will fallback sex='U' and subset=<current split>.", file=sys.stderr)

    # Deterministic processing order
    for subset in CANONICAL_SUBSETS:
        if subset not in cfg.subset_dirs:
            continue

        out_csv = out_dir / f"{subset}.csv"
        if out_csv.exists() and not cfg.overwrite:
            print(f"[SKIP] {out_csv} exists (use --overwrite to regenerate)")
            continue

        subset_root = Path(cfg.subset_dirs[subset]).expanduser().resolve()
        df = build_subset_metadata(
            subset_root=subset_root,
            subset=subset,
            librispeech_root_for_relpath=Path(cfg.librispeech_dir).resolve() if cfg.librispeech_dir else subset_root,
            speakers_map=speakers_map,
            strict=cfg.strict,
        )

        min_samples = int(round(cfg.min_seconds * cfg.rate))
        df = df[df["length"] >= min_samples]
        df = df.sort_values(["length", "origin_path"], ascending=[True, True]).reset_index(drop=True)

        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv} (rows={len(df)})")


def resolve_subsets_under_root(librispeech_dir: Path) -> Dict[str, str]:
    """Return subset -> directory for subsets that exist directly under a unified root."""
    existing_dirs = {p.name: str(p) for p in librispeech_dir.iterdir() if p.is_dir()}
    return {s: existing_dirs[s] for s in CANONICAL_SUBSETS if s in existing_dirs}


def resolve_speakers_txt(
    speakers_txt_arg: Optional[str],
    librispeech_dir: Optional[Path],
    subset_dirs: List[Path],
    max_up_levels: int = 6,
) -> Optional[Path]:
    """
    Resolve SPEAKERS.TXT location with the following priority:
      1) --speakers_txt (explicit override)
      2) <librispeech_dir>/SPEAKERS.TXT (if librispeech_dir is provided)
      3) Auto-detect by walking up from each subset_dir and looking for SPEAKERS.TXT
      4) None
    """
    if speakers_txt_arg:
        p = Path(speakers_txt_arg).expanduser().resolve()
        return p if p.exists() else None

    if librispeech_dir:
        p = librispeech_dir.expanduser().resolve() / "SPEAKERS.TXT"
        if p.exists():
            return p

    for sd in subset_dirs:
        cur = sd.expanduser().resolve()
        for _ in range(max_up_levels + 1):
            candidate = cur / "SPEAKERS.TXT"
            if candidate.exists():
                return candidate
            if cur.parent == cur:
                break
            cur = cur.parent

    return None


def load_speakers_map(speakers_txt: Optional[Path], strict: bool) -> Dict[int, Tuple[str, str]]:
    """
    Parse SPEAKERS.TXT and return a mapping:
        speaker_id -> (sex, subset)

    If SPEAKERS.TXT is unavailable or parsing fails and strict=False, returns an empty mapping.
    Callers should apply a non-empty fallback to avoid null fields in the output.
    """
    if not speakers_txt or not speakers_txt.exists():
        if strict:
            raise FileNotFoundError("SPEAKERS.TXT not found. Provide --speakers_txt or --librispeech_dir.")
        return {}

    try:
        df = pd.read_csv(
            speakers_txt,
            sep="|",
            skiprows=11,
            header=0,
            names=["speaker_ID", "sex", "subset", "minutes", "names"],
            skipinitialspace=True,
            on_bad_lines="skip",
            engine="python",
        )
    except TypeError:
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
    except Exception as e:
        if strict:
            raise
        print(f"[WARN] Failed to parse SPEAKERS.TXT ({e}). Falling back to sex='U', subset=<split>.", file=sys.stderr)
        return {}

    df = df.drop(columns=[c for c in ["minutes", "names"] if c in df.columns], errors="ignore")
    df["sex"] = df["sex"].astype(str).str.strip()
    df["subset"] = df["subset"].astype(str).str.strip()

    # Compatibility fallback used in many recipes.
    try:
        speaker_ids = set(int(x) for x in df["speaker_ID"].dropna().tolist())
    except Exception:
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
            if strict:
                raise
            continue

    return mapping


def build_subset_metadata(
    subset_root: Path,
    subset: str,
    librispeech_root_for_relpath: Path,
    speakers_map: Dict[int, Tuple[str, str]],
    strict: bool,
) -> pd.DataFrame:
    """
    Build utterance-level metadata for a single subset directory.

    `librispeech_root_for_relpath` controls how origin_path is computed:
      - In --all mode: it's the unified LibriSpeech root -> origin_path is relative to the root.
      - In --explicit mode without a root: origin_path is relative to the subset root.

    Output schema:
      speaker_ID, sex, subset, length, origin_path

    Fallback behavior:
      - If SPEAKERS.TXT mapping is unavailable or a speaker is missing, the script writes:
          sex   = "U"
          subset = <current split name> (e.g., "dev-clean")
        This guarantees the columns are never empty in explicit mode.
    """
    if not subset_root.exists():
        msg = f"Subset directory does not exist: {subset_root}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg} (skipping)", file=sys.stderr)
        return pd.DataFrame(columns=["speaker_ID", "sex", "subset", "length", "origin_path"])

    flac_paths = sorted(glob.glob(str(subset_root / "**" / "*.flac"), recursive=True))

    rows: List[dict] = []
    for fp in tqdm(flac_paths, desc=f"scan {subset}", total=len(flac_paths)):
        p = Path(fp)
        try:
            spk_id_str = p.name.split("-")[0]
            spk_id = int(spk_id_str)
        except Exception as e:
            msg = f"Failed to parse speaker ID from filename: {p}"
            if strict:
                raise ValueError(msg) from e
            print(f"[WARN] {msg} (skipping)", file=sys.stderr)
            continue

        try:
            length = int(sf.info(str(p)).frames)
        except Exception as e:
            msg = f"Failed to read audio info: {p}"
            if strict:
                raise RuntimeError(msg) from e
            print(f"[WARN] {msg} (skipping)", file=sys.stderr)
            continue

        if speakers_map and spk_id in speakers_map:
            sex, spk_subset = speakers_map[spk_id]
        else:
            # Hard fallback to avoid empty fields in explicit mode.
            sex, spk_subset = ("U", subset)
            if speakers_map:
                msg = f"Speaker {spk_id} not found in SPEAKERS.TXT mapping for file: {p}"
                if strict:
                    raise KeyError(msg)
                print(f"[WARN] {msg} (fallback sex='U', subset='{subset}')", file=sys.stderr)

        origin_path = os.path.relpath(str(p), str(librispeech_root_for_relpath))

        rows.append(
            {
                "speaker_ID": str(spk_id_str),
                "sex": sex,
                "subset": spk_subset,
                "length": length,
                "origin_path": origin_path,
            }
        )

    return pd.DataFrame(rows, columns=["speaker_ID", "sex", "subset", "length", "origin_path"])


if __name__ == "__main__":
    main()
