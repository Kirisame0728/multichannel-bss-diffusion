#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import torch
import torchaudio


def add_repo_root_to_syspath() -> str:
    """
    Add the repository root to sys.path so imports like `src.*` work when the script
    is executed from baselines/tfgridnet/.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_REPO_ROOT = add_repo_root_to_syspath()

from src.dataloaders.dataset_mix2_test import Mix2TestDataset  # noqa: E402
from src.metrics.sdr import batch_SDR_torch  # noqa: E402
from src.metrics.eval_metrics import sisdr_batch, pesq_batch, estoi_batch  # noqa: E402

# Local model implementation (kept unchanged).
from tfgridnet_separator import TFGridNet  # noqa: E402


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def append_row(csv_path: str, row: Dict[str, Any]) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def save_sources(est: torch.Tensor, out_dir: str, utt_id: str, sr: int) -> None:
    """
    Save separated sources.

    est: (K, T)
    Output layout:
      out_dir/<utt_id>/s1.wav, s2.wav, ...
    """
    utt_dir = os.path.join(out_dir, str(utt_id))
    ensure_dir(utt_dir)
    k = int(est.shape[0])
    for i in range(k):
        wav_path = os.path.join(utt_dir, f"s{i+1}.wav")
        torchaudio.save(wav_path, est[i].unsqueeze(0).cpu(), sample_rate=sr)


def load_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load model parameters from checkpoint.

    Supported conventions:
      - {"model": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - state_dict itself
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # fall back: treat dict as a raw state_dict
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


@torch.no_grad()
def separate_with_tfgridnet(model: TFGridNet, mixture_ct: torch.Tensor) -> torch.Tensor:
    """
    Run TF-GridNet separation.

    mixture_ct: (C, T)
    return: (K, T)
    """
    if mixture_ct.dim() != 2:
        raise ValueError(f"Expected mixture shape (C,T), got {tuple(mixture_ct.shape)}")

    c, t = mixture_ct.shape
    mix_btc = mixture_ct.transpose(0, 1).unsqueeze(0)  # (1, T, C)
    ilens = torch.tensor([t], device=mix_btc.device)   # (1,)

    ys, _, _ = model(mix_btc, ilens)  # list length K, each (1, T)
    est = torch.stack([y.squeeze(0) for y in ys], dim=0)  # (K, T)
    return est


def get_item_fields(item: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Normalize dataset output into (mixture, early, tail, utt_id).
    Supports dict style or tuple/list style (mixture, early, tail, utt_id).
    """
    if isinstance(item, dict):
        mixture = item["mixture"]
        early = item["early"]
        tail = item["tail"]
        utt_id = str(item.get("utt_id", "unknown"))
        return mixture, early, tail, utt_id

    if isinstance(item, (tuple, list)) and len(item) == 4:
        mixture, early, tail, utt_id = item
        return mixture, early, tail, str(utt_id)

    raise TypeError(f"Unsupported dataset item type/format: {type(item)}")


def mean_finite(xs: List[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size > 0 else float("nan")


def summarize(values: List[float]) -> Dict[str, float]:
    """
    Match IVA summary.csv statistics:
      count, mean, median, std (sample std with ddof=1; std=NaN if count==1)
    """
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0.0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    if a.size == 1:
        return {"count": float(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float("nan")}
    return {
        "count": float(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)),
    }


def scale_to_ref(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Least-squares scaling of estimated sources to reference sources, per source.

    est/ref: (K, T)
    return: scaled_est (K, T)
    """
    num = torch.sum(ref * est, dim=1, keepdim=True)       # (K,1)
    den = torch.sum(est * est, dim=1, keepdim=True) + eps # (K,1)
    alpha = num / den                                     # (K,1)
    return est * alpha


def main() -> None:
    parser = argparse.ArgumentParser("TF-GridNet baseline: inference on Mix2TestDataset")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--n_channels", type=int, default=6)
    parser.add_argument("--select_channels", type=str, default="0,2,4")

    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")

    parser.add_argument("--compute_pesq", action="store_true")
    parser.add_argument("--compute_estoi", action="store_true")
    parser.add_argument("--strict_perceptual", action="store_true")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    select_channels = parse_int_list(args.select_channels)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ensure_dir(args.save_dir)
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

    # Match training-time microphone count after channel selection.
    effective_mics = len(select_channels) if select_channels is not None else args.n_channels

    model = TFGridNet(
        input_dim=None,
        n_srcs=args.num_speakers,
        n_fft=256,
        stride=64,
        window="hann",
        n_imics=effective_mics,
        attn_n_head=4,
    ).to(device)

    state = load_state_dict(args.ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    sdr_list: List[float] = []
    sisdr_list: List[float] = []
    pesq_list: List[float] = []
    estoi_list: List[float] = []
    mixsnr_list: List[float] = []

    end_idx = min(len(dataset), args.start_sample + args.n_samples)
    for i in range(args.start_sample, end_idx):
        mixture, early, tail, utt_id = get_item_fields(dataset[i])

        if args.skip_existing:
            utt_dir = os.path.join(args.save_dir, utt_id)
            ok = os.path.isdir(utt_dir) and all(
                os.path.exists(os.path.join(utt_dir, f"s{k+1}.wav")) for k in range(args.num_speakers)
            )
            if ok:
                continue

        sources = early + tail  # (K, C, T)

        t = min(int(mixture.shape[-1]), int(sources.shape[-1]))
        mixture = mixture[:, :t]
        sources = sources[:, :, :t]

        mixture = mixture.to(device)
        est = separate_with_tfgridnet(model, mixture).detach().cpu()  # (K, T)

        save_sources(est, args.save_dir, utt_id, sr=args.sample_rate)

        # Metric reference microphone follows the same convention as IVA baseline: channel 0.
        ref = sources[:, 0, :].detach().cpu()  # (K, T)

        # SDR implementation expects (B, K, T). Keep scaling for SDR path only.
        est_for_sdr = scale_to_ref(est, ref)
        sdr_val = float(batch_SDR_torch(est_for_sdr.unsqueeze(0), ref.unsqueeze(0)).item())

        sisdr_val = float(sisdr_batch(est, ref))

        pesq_val = float("nan")
        estoi_val = float("nan")

        if args.compute_pesq:
            try:
                pesq_val = float(pesq_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise
                pesq_val = float("nan")

        if args.compute_estoi:
            try:
                estoi_val = float(estoi_batch(est.numpy(), ref.numpy(), sr=args.sample_rate))
            except Exception:
                if args.strict_perceptual:
                    raise
                estoi_val = float("nan")

        mix_snr = float(
            10.0 * torch.log10(
                (mixture[0].detach().cpu().pow(2).sum() + 1e-8)
                / ((mixture[0].detach().cpu() - est.sum(dim=0)).pow(2).sum() + 1e-8)
            ).item()
        )

        append_row(
            per_utt_csv,
            {
                "utt_id": utt_id,
                "trial_0_sdr": sdr_val,
                "trial_0_sisdr": sisdr_val,
                "trial_0_pesq": pesq_val,
                "trial_0_estoi": estoi_val,
                "mix_snr_refmic": mix_snr,
            },
        )

        sdr_list.append(sdr_val)
        sisdr_list.append(sisdr_val)
        pesq_list.append(pesq_val)
        estoi_list.append(estoi_val)
        mixsnr_list.append(mix_snr)

        if args.verbose:
            print(
                f"[TFGridNet] utt_id={utt_id} idx={i} "
                f"SDR={sdr_val:.3f} SI-SDR={sisdr_val:.3f} "
                f"PESQ={pesq_val:.3f} eSTOI={estoi_val:.3f} MixSNR={mix_snr:.2f}"
            )

    # Write summary.csv in the same format as IVA baseline.
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "median", "std"])

        for name, values in [
            ("sdr", sdr_list),
            ("sisdr", sisdr_list),
            ("pesq", pesq_list),
            ("estoi", estoi_list),
            ("mixsnr", mixsnr_list),
        ]:
            s = summarize(values)
            if int(s["count"]) == 0:
                continue
            w.writerow([name, int(s["count"]), s["mean"], s["median"], s["std"]])

    print(f"Wrote: {per_utt_csv}")
    print(f"Wrote: {summary_csv}")


if __name__ == "__main__":
    main()