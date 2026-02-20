#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torchaudio

# add repo root
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.dataloaders.dataset_mix2_test import Mix2TestDataset  # noqa
from src.metrics.sdr import batch_SDR_torch  # noqa
from src.metrics.eval_metrics import sisdr_batch, pesq_batch, estoi_batch  # noqa

from encoder import Encoder
from decoder import Decoder
from separate import finetune


def parse_int_list(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_ckpt(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def summarize(values: List[float]):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=1)) if a.size > 1 else float("nan"))


def scale_to_ref(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # scale est to best match ref (per source)
    num = torch.sum(ref * est, dim=1, keepdim=True)
    den = torch.sum(est * est, dim=1, keepdim=True) + eps
    return est * (num / den)


def main():
    p = argparse.ArgumentParser("Neural-FCA baseline: inference")
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--sample_rate", type=int, default=8000)
    p.add_argument("--num_speakers", type=int, default=2)
    p.add_argument("--n_channels", type=int, default=6)
    p.add_argument("--select_channels", type=str, default="0,2,4")

    p.add_argument("--start_sample", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--skip_existing", action="store_true")

    p.add_argument("--compute_pesq", action="store_true")
    p.add_argument("--compute_estoi", action="store_true")
    p.add_argument("--strict_perceptual", action="store_true")

    # finetune hyperparams (pass to separate.py)
    p.add_argument("--n_iter", type=int, default=10)
    p.add_argument("--n_ziter", type=int, default=20)
    p.add_argument("--n_hiter", type=int, default=1)
    p.add_argument("--out_ch", type=int, default=0)

    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    ensure_dir(args.save_dir)
    per_csv = os.path.join(args.save_dir, "per_utt.csv")
    sum_csv = os.path.join(args.save_dir, "summary.csv")

    select_channels = parse_int_list(args.select_channels)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # dataset
    ds = Mix2TestDataset(
        root_dir=args.root_dir,
        split=args.split,
        sample_rate=args.sample_rate,
        n_src=args.num_speakers,
        num_channels=args.n_channels,
        select_channels=select_channels,
    )

    # model
    # enc = Encoder(K=args.num_speakers).to(device).eval()
    # dec = Decoder(K=args.num_speakers).to(device).eval()
    F = args.n_fft // 2 + 1
    select_channels = parse_int_list(args.select_channels)
    M = len(select_channels) if select_channels is not None else args.n_channels
    enc = Encoder(F=F, M=M, K=args.num_speakers).to(device).eval()
    dec = Decoder(F=F, K=args.num_speakers).to(device).eval()

    state = load_ckpt(args.ckpt)
    enc.load_state_dict(state["enc"], strict=True)
    dec.load_state_dict(state["dec"], strict=True)

    sdrs, sisdrs, pesqs, estois = [], [], [], []

    end = min(len(ds), args.start_sample + args.n_samples)
    for idx in range(args.start_sample, end):
        item = ds[idx]
        # Support both tuple/list and dict dataset return
        if isinstance(item, dict):
            mixture = item["mixture"]; early = item["early"]; tail = item["tail"]; utt_id = str(item.get("utt_id", idx))
        else:
            mixture, early, tail, utt_id = item
            utt_id = str(utt_id)

        utt_dir = os.path.join(args.save_dir, utt_id)
        if args.skip_existing and os.path.isdir(utt_dir):
            ok = all(os.path.exists(os.path.join(utt_dir, f"s{k+1}.wav")) for k in range(args.num_speakers))
            if ok:
                continue

        # mixture: (C,T)
        sources = early + tail                 # (K,C,T)
        T = min(int(mixture.shape[-1]), int(sources.shape[-1]))
        mixture = mixture[:, :T]
        sources = sources[:, :, :T]
        ref = sources[:, 0, :].cpu()          # reference mic

        # separate.py uses librosa; pass numpy mix (C,T)
        est_np = finetune(
            mixture.cpu().numpy(),
            enc,
            dec,
            n_iter=args.n_iter,
            n_ziter=args.n_ziter,
            n_hiter=args.n_hiter,
            out_ch=args.out_ch,
        )  # (K,T) numpy
        est = torch.from_numpy(np.asarray(est_np)).float().cpu()

        ensure_dir(utt_dir)
        for k in range(args.num_speakers):
            torchaudio.save(os.path.join(utt_dir, f"s{k+1}.wav"), est[k].unsqueeze(0), sample_rate=args.sample_rate)

        est_sdr = scale_to_ref(est, ref)
        sdr = float(batch_SDR_torch(est_sdr.unsqueeze(0), ref.unsqueeze(0)).item())
        sisdr = float(sisdr_batch(est, ref))

        pesq = float("nan")
        estoi = float("nan")
        if args.compute_pesq:
            try:
                pesq = float(pesq_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise
        if args.compute_estoi:
            try:
                estoi = float(estoi_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise

        append_row(per_csv, {
            "utt_id": utt_id,
            "trial_0_sdr": sdr,
            "trial_0_sisdr": sisdr,
            "trial_0_pesq": pesq,
            "trial_0_estoi": estoi,
        })

        sdrs.append(sdr); sisdrs.append(sisdr); pesqs.append(pesq); estois.append(estoi)

        if args.verbose:
            print(f"[NeuralFCA] {utt_id} idx={idx} SDR={sdr:.3f} SI-SDR={sisdr:.3f} PESQ={pesq:.3f} eSTOI={estoi:.3f}")

    # summary
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "median", "std"])
        for name, arr in [("sdr", sdrs), ("sisdr", sisdrs), ("pesq", pesqs), ("estoi", estois)]:
            s = summarize(arr)
            if s is None:
                continue
            w.writerow([name, s["count"], s["mean"], s["median"], s["std"]])

    print(f"Wrote: {per_csv}")
    print(f"Wrote: {sum_csv}")


if __name__ == "__main__":
    main()