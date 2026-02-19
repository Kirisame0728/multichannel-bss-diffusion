#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from typing import Dict, Optional, List

import numpy as np
import torch
import torchaudio


def _add_repo_root_to_syspath() -> str:
    """
    Add repository root to sys.path so this script can be run from baselines/tfgridnet/.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_REPO_ROOT = _add_repo_root_to_syspath()

from src.dataloaders.dataset_mix2_test import Mix2TestDataset  # type: ignore
from src.metrics.sdr import batch_SDR_torch  # type: ignore
from src.metrics.eval_metrics import sisdr_batch, pesq_batch, estoi_batch  # type: ignore

# IMPORTANT: import the original class name from your unchanged tfgridnet_separator.py
from tfgridnet_separator import TFGridNet  # noqa: E402


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@torch.no_grad()
def _save_sources(est: torch.Tensor, out_dir: str, utt_id: str, sr: int) -> None:
    """
    est: (K, T)
    Save to out_dir/<utt_id>/s1.wav, s2.wav, ...
    """
    utt_dir = os.path.join(out_dir, str(utt_id))
    _ensure_dir(utt_dir)
    K = int(est.shape[0])
    for k in range(K):
        torchaudio.save(
            os.path.join(utt_dir, f"s{k+1}.wav"),
            est[k].unsqueeze(0).cpu(),
            sample_rate=sr,
        )


def _append_row(csv_path: str, row: Dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


@torch.no_grad()
def _separate_tfgridnet(
    model: TFGridNet,
    mixture_ct: torch.Tensor,   # (C, T)
) -> torch.Tensor:
    """
    Run TFGridNet forward.

    TFGridNet expects input shape (B, N, M) where:
      N = number of time samples, M = number of channels (microphones).

    Returns:
      est_kt: (K, T)
    """
    if mixture_ct.dim() != 2:
        raise ValueError(f"Expected mixture shape (C,T), got {tuple(mixture_ct.shape)}")

    C, T = int(mixture_ct.shape[0]), int(mixture_ct.shape[1])
    mixture_bnm = mixture_ct.transpose(0, 1).unsqueeze(0).contiguous()  # (1, T, C)
    ilens = torch.tensor([T], dtype=torch.long, device=mixture_bnm.device)

    outs_list, _, _ = model(mixture_bnm, ilens)  # List[(1,T)] length K
    est_bkt = torch.stack(outs_list, dim=1)      # (1,K,T)
    return est_bkt.squeeze(0)                    # (K,T)


def main() -> None:
    parser = argparse.ArgumentParser("TF-GridNet baseline: separation + evaluation on Mix2TestDataset")

    # Paths
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for wavs/csv")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt/.pth)")

    # Dataset
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--n_channels", type=int, default=6)
    parser.add_argument("--select_channels", type=str, default="0,2,4", help='e.g. "0,2,4"')

    # Eval control
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")

    # Metrics
    parser.add_argument("--compute_pesq", action="store_true")
    parser.add_argument("--compute_estoi", action="store_true")
    parser.add_argument("--strict_perceptual", action="store_true")

    # Runtime
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    select_channels = None
    if args.select_channels is not None and args.select_channels.strip() != "":
        select_channels = [int(x) for x in args.select_channels.split(",") if x.strip() != ""]

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    _ensure_dir(args.save_dir)
    per_utt_csv = os.path.join(args.save_dir, "per_utt.csv")
    summary_csv = os.path.join(args.save_dir, "summary.csv")

    dataset = Mix2TestDataset(
        root_dir=args.root_dir,
        split=args.split,
        sample_rate=args.sample_rate,
        n_src=args.num_speakers,
        num_channels=args.n_channels,
        select_channels=select_channels,
    )

    # Build original TFGridNet (do NOT change tfgridnet_separator.py)
    model = TFGridNet(
        input_dim=None,
        n_srcs=args.num_speakers,
        n_fft=512,
        stride=128,
        window="hann",
        n_imics=len(select_channels) if select_channels is not None else args.n_channels,
        n_layers=6,
        ref_channel=-1,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    sdr_list, sisdr_list, pesq_list, estoi_list = [], [], [], []
    mixsnr_list = []

    end_idx = min(len(dataset), args.start_sample + args.n_samples)
    for i in range(args.start_sample, end_idx):
        item = dataset[i]
        if isinstance(item, dict):
            utt_id = item.get("utt_id", f"sample_{i}")
            mixture = item["mixture"]         # (C, T)
            early = item.get("early", None)   # (K, C, T)
            tail = item.get("tail", None)     # (K, C, T)
        else:
            mixture, early, tail, utt_id = item

        utt_id = str(utt_id)

        if args.skip_existing:
            utt_dir = os.path.join(args.save_dir, utt_id)
            ok = os.path.isdir(utt_dir) and all(
                os.path.exists(os.path.join(utt_dir, f"s{k+1}.wav")) for k in range(args.num_speakers)
            )
            if ok:
                continue

        if early is None or tail is None:
            raise KeyError("Mix2TestDataset item must provide 'early' and 'tail' for evaluation.")
        sources = early + tail  # (K, C, T)

        T = min(int(mixture.shape[-1]), int(sources.shape[-1]))
        mixture = mixture[:, :T]
        sources = sources[:, :, :T]

        mixture = mixture.to(device)

        est = _separate_tfgridnet(model, mixture).detach().cpu()  # (K, T)
        _save_sources(est, args.save_dir, utt_id, sr=args.sample_rate)

        # Keep the same evaluation convention as your IVA baseline: ref mic = ch0
        ref = sources[:, 0, :].detach().cpu()  # (K, T)

        sdr_tmp = batch_SDR_torch(est, ref)
        sdr_val = float(sdr_tmp.item()) if torch.is_tensor(sdr_tmp) else float(sdr_tmp)
        sisdr_val = float(sisdr_batch(est, ref))

        pesq_val = float("nan")
        estoi_val = float("nan")

        if args.compute_pesq:
            try:
                pesq_val = float(pesq_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise

        if args.compute_estoi:
            try:
                estoi_val = float(estoi_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise

        mix_snr = float(
            10.0
            * torch.log10(
                (mixture[0].detach().cpu().pow(2).sum() + 1e-8)
                / ((mixture[0].detach().cpu() - est.sum(dim=0)).pow(2).sum() + 1e-8)
            ).item()
        )

        row = {
            "utt_id": utt_id,
            "trial_0_sdr": sdr_val,
            "trial_0_sisdr": sisdr_val,
            "trial_0_pesq": pesq_val,
            "trial_0_estoi": estoi_val,
            "mix_snr_refmic": mix_snr,
        }
        _append_row(per_utt_csv, row)

        sdr_list.append(sdr_val)
        sisdr_list.append(sisdr_val)
        if np.isfinite(pesq_val):
            pesq_list.append(pesq_val)
        if np.isfinite(estoi_val):
            estoi_list.append(estoi_val)
        mixsnr_list.append(mix_snr)

        if args.verbose:
            print(
                f"[TFGridNet] utt_id={utt_id} idx={i} "
                f"SDR={sdr_val:.3f} SI-SDR={sisdr_val:.3f} "
                f"PESQ={pesq_val:.3f} eSTOI={estoi_val:.3f} MixSNR={mix_snr:.2f}"
            )

    def _mean(xs: List[float]) -> float:
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if len(xs) > 0 else float("nan")

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean"])
        w.writerow(["SDR", _mean(sdr_list)])
        w.writerow(["SI-SDR", _mean(sisdr_list)])
        w.writerow(["PESQ", _mean(pesq_list)])
        w.writerow(["eSTOI", _mean(estoi_list)])
        w.writerow(["MixSNR", _mean(mixsnr_list)])

    print(f"Wrote: {per_utt_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()
