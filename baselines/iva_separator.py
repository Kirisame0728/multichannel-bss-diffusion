#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
import importlib.util
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchaudio


# ============================================================
# Repo bootstrap (so running inside baselines/ works)
# ============================================================
def _repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def _load_module_from_path(mod_name: str, file_path: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create spec for {mod_name} from {file_path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[attr-defined]
    return m


REPO_ROOT = _repo_root()

# Load metrics exactly from your src/metrics/{sdr.py, eval_metrics.py}
METRICS_DIR = os.path.join(REPO_ROOT, "src", "metrics")
SDR_PATH = os.path.join(METRICS_DIR, "sdr.py")
EVAL_METRICS_PATH = os.path.join(METRICS_DIR, "eval_metrics.py")

sdr_mod = _load_module_from_path("mix2_sdr", SDR_PATH)
eval_mod = _load_module_from_path("mix2_eval_metrics", EVAL_METRICS_PATH)

batch_SDR_torch = getattr(sdr_mod, "batch_SDR_torch")
sisdr_batch = getattr(eval_mod, "sisdr_batch")
pesq_batch = getattr(eval_mod, "pesq_batch")
estoi_batch = getattr(eval_mod, "estoi_batch")

# Load dataset from src/dataloaders/dataset_mix2_test.py
DATALOADERS_DIR = os.path.join(REPO_ROOT, "src", "dataloaders")
DATASET_PATH = os.path.join(DATALOADERS_DIR, "dataset_mix2_test.py")
ds_mod = _load_module_from_path("mix2_dataset_test", DATASET_PATH)
Mix2TestDataset = getattr(ds_mod, "Mix2TestDataset")


# ============================================================
# CSV / stats utilities (aligned to your original)
# ============================================================
def append_per_utt_row(csv_path: str, row: dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def summarize(values: List[float]) -> Dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
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


def save_sdr_list(sdr_list, mix_snr_list, save_dir, max_trials) -> None:
    sdr_csv_path = os.path.join(save_dir, "sdr_mix_snr.csv")
    with open(sdr_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        title = (
            ["Sample Index"]
            + [f"SDR{trial_idx}" for trial_idx in range(max_trials)]
            + [f"Mix SNR {trial_idx}" for trial_idx in range(max_trials)]
            + ["max_sdr", "max_mix_snr"]
        )
        writer.writerow(title)

        for (sample_index, sdr_values), (_, mix_snr_values) in zip(sdr_list, mix_snr_list):
            sdr_values = sdr_values + [np.nan for _ in range(max_trials - len(sdr_values))]
            mix_snr_values = mix_snr_values + [np.nan for _ in range(max_trials - len(mix_snr_values))]
            max_sdr = np.nanmax(sdr_values)
            max_mix_snr = np.nanmax(mix_snr_values)
            writer.writerow([sample_index] + sdr_values + mix_snr_values + [max_sdr, max_mix_snr])


def save_separated_samples(outs: torch.Tensor, save_dir: str, utt_id: str, sr: int) -> None:
    """
    Save separated waveforms to:
      <save_dir>/<utt_id>/s1.wav, s2.wav, ...
    outs: (n_src, T)
    """
    sample_dir = os.path.join(save_dir, str(utt_id))
    os.makedirs(sample_dir, exist_ok=True)

    n_src, _ = outs.shape
    for k in range(n_src):
        path = os.path.join(sample_dir, f"s{k+1}.wav")
        torchaudio.save(path, outs[k].unsqueeze(0).cpu(), sample_rate=sr)


def check_existing_outputs(save_dir: str, utt_id: str, num_speakers: int) -> bool:
    trial_dir = os.path.join(save_dir, str(utt_id))
    if not os.path.exists(trial_dir):
        return False
    expected = [f"s{k+1}.wav" for k in range(num_speakers)]
    present = set(os.listdir(trial_dir))
    return all(x in present for x in expected)


def calc_snr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    SNR = 10*log10( ||ref||^2 / ||ref-est||^2 )
    ref/est: (T,)
    """
    ref = ref.float()
    est = est.float()
    num = torch.sum(ref * ref) + eps
    den = torch.sum((ref - est) ** 2) + eps
    return 10.0 * torch.log10(num / den)


# ============================================================
# STFT / projection-back / ISTFT  (same as your original)
# ============================================================
def stft_multich(
    mixture: torch.Tensor,  # (C, T)
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-channel STFT.
    Returns:
      X_ftc: (F, TT, C) complex
      window: Hann window tensor
    """
    mixture = mixture.detach().float().cpu()
    window = torch.hann_window(win_length, periodic=True)

    X_list = []
    for c in range(mixture.shape[0]):
        Xc = torch.stft(
            mixture[c],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )  # (F, TT)
        X_list.append(Xc)
    X_ftc = torch.stack(X_list, dim=-1)  # (F, TT, C)
    return X_ftc, window


def projection_back(Y_ftk: torch.Tensor, X_ref_ft: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Projection-back scaling to match reference microphone.
    Y_ftk: (F, TT, K)
    X_ref_ft: (F, TT)
    """
    num = torch.sum(X_ref_ft.unsqueeze(-1) * torch.conj(Y_ftk), dim=1)  # (F, K)
    den = torch.sum(torch.abs(Y_ftk) ** 2, dim=1) + eps                # (F, K)
    alpha = num / den                                                  # (F, K)
    return Y_ftk * alpha.unsqueeze(1)                                  # (F, TT, K)


def istft_sources(
    Y_ftk: torch.Tensor,  # (F, TT, K)
    window: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    length: int,
) -> torch.Tensor:
    """
    ISTFT for multiple separated sources.
    Returns:
      (K, T) float
    """
    K = Y_ftk.shape[-1]
    outs = []
    for k in range(K):
        y = torch.istft(
            Y_ftk[:, :, k],
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            length=length,
        )
        outs.append(y.float())
    return torch.stack(outs, dim=0)


# ============================================================
# TorchIVA AuxIVA-IP wrapper (same behavior as original)
# ============================================================
def torchiva_auxiva_ip_separate(
    X_ftc: torch.Tensor,  # (F, TT, C)
    n_src: int,
    n_iter: int,
) -> torch.Tensor:
    """
    Run AuxIVA-IP using torchiva.AuxIVA_IP.
    Returns: (F, TT, K)
    """
    import torchiva

    if not hasattr(torchiva, "AuxIVA_IP"):
        raise RuntimeError("torchiva.AuxIVA_IP is not available. Install/verify torchiva.")

    C = X_ftc.shape[-1]
    F = X_ftc.shape[0]
    TT = X_ftc.shape[1]

    X_cft = X_ftc.permute(2, 0, 1).contiguous()  # (C, F, TT)

    sep = torchiva.AuxIVA_IP(n_iter=n_iter, n_src=n_src)
    Y = sep(X_cft)

    if not torch.is_tensor(Y) or Y.dim() != 3:
        raise RuntimeError(f"AuxIVA_IP returned unexpected type/shape: type={type(Y)}, shape={getattr(Y, 'shape', None)}")

    shp = tuple(Y.shape)

    # (K, F, TT) -> (F, TT, K)
    if shp[0] == n_src and shp[1] == F and shp[2] == TT:
        return Y.permute(1, 2, 0).contiguous()

    # (F, TT, K)
    if shp[0] == F and shp[1] == TT and shp[2] == n_src:
        return Y.contiguous()

    # (F, K, TT) -> (F, TT, K)
    if shp[0] == F and shp[1] == n_src and shp[2] == TT:
        return Y.permute(0, 2, 1).contiguous()

    raise RuntimeError(
        "AuxIVA_IP output layout unexpected.\n"
        f"Expected one of: (K,F,TT)=({n_src},{F},{TT}), (F,TT,K)=({F},{TT},{n_src}), (F,K,TT)=({F},{n_src},{TT}).\n"
        f"Got: {shp}\n"
        "Adjust torchiva_auxiva_ip_separate() for this torchiva version."
    )


@torch.no_grad()
def iva_separate_waveform_refmic(
    mixture: torch.Tensor,  # (C, T)
    n_src: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
    ref_mic: int = 0,
) -> torch.Tensor:
    """
    STFT -> AuxIVA-IP -> projection-back -> ISTFT.
    Returns: (n_src, T) on CPU, scaled to reference microphone.
    """
    C, T = mixture.shape
    if not (0 <= ref_mic < C):
        raise ValueError(f"ref_mic {ref_mic} out of range for C={C}")
    if n_src > C:
        raise ValueError(f"n_src {n_src} cannot exceed C {C}")

    X_ftc, window = stft_multich(mixture, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    Y_ftk = torchiva_auxiva_ip_separate(X_ftc, n_src=n_src, n_iter=n_iter)

    X_ref = X_ftc[:, :, ref_mic]
    Y_ftk = projection_back(Y_ftk, X_ref)

    est = istft_sources(
        Y_ftk,
        window=window,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        length=T,
    )
    return est


# ============================================================
# Dataset builder (adapted to your current src/dataloaders)
# ============================================================
def _parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.replace(" ", ",").split(",") if x != ""]


def build_dataset(args):
    return Mix2TestDataset(
        root_dir=args.root_dir,
        split=args.split,
        sample_rate=args.sample_rate,
        n_src=args.num_speakers,
        num_channels=args.n_channels,
        select_channels=args.select_channels if len(args.select_channels) > 0 else None,
        mixture_folder=args.mixture_folder,
        early_folder=args.early_folder,
        tail_folder=args.tail_folder,
        use_mixture_file=args.use_mixture_file,
    )


# ============================================================
# Main evaluation loop (printing format restored)
# ============================================================
def run(dataset, args):
    os.makedirs(args.save_dir, exist_ok=True)

    per_utt_path = os.path.join(args.save_dir, "per_utt.csv")
    summary_path = os.path.join(args.save_dir, "summary.csv")

    agg = {"sdr": [], "sisdr": [], "pesq": [], "estoi": []}
    sdr_list = []
    mixsnr_list = []
    max_trials = 1

    for i, (mixture, early, tail, utt_id) in enumerate(dataset):
        if i < args.start_sample:
            continue
        if (i - args.start_sample) >= args.n_samples:
            break

        utt_id = str(utt_id)

        if args.skip_existing and check_existing_outputs(args.save_dir, utt_id, args.num_speakers):
            continue

        sources = early + tail  # (K, C, T)

        # Align lengths
        T = min(mixture.shape[-1], sources.shape[-1])
        mixture = mixture[:, :T]
        sources = sources[:, :, :T]

        # IVA separation
        est = iva_separate_waveform_refmic(
            mixture=mixture,
            n_src=args.num_speakers,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            n_iter=args.iva_iter,
            ref_mic=args.sep_ref_mic,
        )  # (K, T)

        save_separated_samples(est, args.save_dir, utt_id, sr=args.sample_rate)

        # ---- metrics: strictly same as your original three-file baseline ----
        est_ch0 = est.detach().cpu()
        ref_ch0 = sources[:, 0, :].detach().cpu()

        sdr = batch_SDR_torch(est_ch0.unsqueeze(0), ref_ch0.unsqueeze(0))
        sdr_val = float(sdr.item())

        mix_snr = float(calc_snr(mixture[0].cpu(), est_ch0.sum(dim=0)).item())

        sisdr_val = float(sisdr_batch(est_ch0, ref_ch0))
        pesq_val = float(pesq_batch(est_ch0.numpy(), ref_ch0.numpy(), sr=args.sample_rate))
        estoi_val = float(estoi_batch(est_ch0.numpy(), ref_ch0.numpy(), sr=args.sample_rate))

        row = {
            "utt_id": utt_id,
            "num_trials": 1,
            "trial_0_sdr": sdr_val,
            "trial_0_sisdr": sisdr_val,
            "trial_0_pesq": pesq_val,
            "trial_0_estoi": estoi_val,
        }
        append_per_utt_row(per_utt_path, row)

        agg["sdr"].append(sdr_val)
        agg["sisdr"].append(sisdr_val)
        agg["pesq"].append(pesq_val)
        agg["estoi"].append(estoi_val)

        sdr_list.append((i, [sdr_val]))
        mixsnr_list.append((i, [mix_snr]))

        # ---- printing format restored (your requested style) ----
        print(
            f"[IVA] utt_id={utt_id} idx={i} "
            f"SDR={sdr_val:.3f} SI-SDR={sisdr_val:.3f} "
            f"PESQ={pesq_val:.3f} eSTOI={estoi_val:.3f} MixSNR={mix_snr:.2f}"
        )

    save_sdr_list(sdr_list, mixsnr_list, args.save_dir, max_trials)

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "median", "std"])
        for m in ["sdr", "sisdr", "pesq", "estoi"]:
            s = summarize(agg[m])
            if int(s["count"]) == 0:
                continue
            w.writerow([m, int(s["count"]), s["mean"], s["median"], s["std"]])


def main():
    parser = argparse.ArgumentParser("IVA baseline (torchiva AuxIVA_IP) aligned to legacy metrics")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--n_channels", type=int, default=6)
    parser.add_argument("--num_speakers", type=int, default=2)

    parser.add_argument("--select_channels", type=str, default="", help='e.g. "0,2,4" (handled by dataset)')

    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=2048)
    parser.add_argument("--iva_iter", type=int, default=100)

    parser.add_argument("--sep_ref_mic", type=int, default=0)

    parser.add_argument("--mixture_folder", type=str, default="observation", choices=["observation", "mix"])
    parser.add_argument("--early_folder", type=str, default="early")
    parser.add_argument("--tail_folder", type=str, default="tail")
    parser.add_argument("--use_mixture_file", action="store_true")

    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--n_samples", type=int, default=999999)
    parser.add_argument("--start_sample", type=int, default=0)

    args = parser.parse_args()

    args.select_channels = _parse_int_list(args.select_channels)

    dataset = build_dataset(args)
    run(dataset, args)


if __name__ == "__main__":
    main()

