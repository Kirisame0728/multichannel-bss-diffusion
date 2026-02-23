#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import csv
import argparse
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# add repo root
_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.dataloaders.dataset_mix2_train import Mix2TrainDataset  # noqa

from encoder import Encoder
from decoder import Decoder
from fca_core import init_H, update_H_em, nll_gaussian


def _parse_int_list(s: str) -> Optional[List[int]]:
    s = (s or "").strip()
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


class UnsupervisedMixAdapter(torch.utils.data.Dataset):
    """
    Wrap Mix2TrainDataset but return only mixture.
    """
    def __init__(self, **kwargs):
        self.ds = Mix2TrainDataset(**kwargs)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        mix, _tgt, utt_id = self.ds[idx]   # ignore tgt
        mix_tc = mix.transpose(0, 1).contiguous()  # (T,C)
        return mix_tc, int(mix_tc.shape[0]), str(utt_id)


def pad_collate_unsup(batch):
    mixes, ilens, utt_ids = zip(*batch)
    B = len(mixes)
    T_max = max(int(x.shape[0]) for x in mixes)
    C = int(mixes[0].shape[1])
    out = mixes[0].new_zeros((B, T_max, C))
    for b in range(B):
        t = int(mixes[b].shape[0])
        out[b, :t, :] = mixes[b]
    return out, torch.tensor(ilens, dtype=torch.long), list(utt_ids)


def cyclic_beta(epoch: int, cycle: int = 10, ratio: float = 0.5, max_beta: float = 1.0) -> float:
    if cycle <= 0:
        return max_beta
    e = (epoch - 1) % cycle
    ramp = max(1, int(round(cycle * ratio)))
    if e >= ramp:
        return max_beta
    return max_beta * float(e + 1) / float(ramp)


def stft_mc(mix_btc: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """
    mix_btc: (B,T,C) real -> (B,T',F,C) complex
    """
    B, T, C = mix_btc.shape
    X = []
    for ch in range(C):
        x = torch.stft(
            mix_btc[:, :, ch],
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
        )  # (B,F,T')
        X.append(x)
    X = torch.stack(X, dim=-1)             # (B,F,T',C)
    return X.permute(0, 2, 1, 3).contiguous()  # (B,T',F,C)


def batch_to_xx(x_btfm: torch.Tensor) -> torch.Tensor:
    return x_btfm[:, :, :, :, None] * x_btfm[:, :, :, None, :].conj()  # (B,T,F,M,M)


def main():
    p = argparse.ArgumentParser("Neural-FCA (unsupervised) trainer")
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--exp_dir", type=str, required=True)

    p.add_argument("--sample_rate", type=int, default=8000)
    p.add_argument("--n_src", type=int, default=2)
    p.add_argument("--target_mode", type=str, default="early+late", choices=["early", "early+late"])
    p.add_argument("--select_channels", type=str, default="0,2,4")
    p.add_argument("--max_len", type=int, default=32000)

    p.add_argument("--batch_size", type=int, default=1)   # baseline default
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)

    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop", type=int, default=256)

    p.add_argument("--n_hiter", type=int, default=5, help="H EM updates per step")

    p.add_argument("--kl_cycle", type=int, default=10)
    p.add_argument("--kl_ratio", type=float, default=0.5)
    p.add_argument("--kl_max_beta", type=float, default=1.0)

    p.add_argument("--kl_max_beta_first", type=float, default=10.0,
                   help="max KL beta for early training (paper: 10.0 for first 50 epochs)")
    p.add_argument("--kl_first_epochs", type=int, default=50,
                   help="number of epochs to use kl_max_beta_first (paper: 50)")

    p.add_argument("--log_interval", type=int, default=20)

    p.add_argument("--limit", type=int, default=-1, help="use only first N training utterances; -1=all")
    args = p.parse_args()

    os.makedirs(args.exp_dir, exist_ok=True)

    select_channels = _parse_int_list(args.select_channels)
    max_len = None if args.max_len <= 0 else args.max_len

    limit = None if args.limit is None or args.limit <= 0 else int(args.limit)
    train_set = UnsupervisedMixAdapter(
        root_dir=args.root_dir,
        split=args.train_split,
        n_src=args.n_src,
        target_mode=args.target_mode,
        sample_rate=args.sample_rate,
        num_channels=(len(select_channels) if select_channels else 6),
        select_channels=select_channels,
        max_len=max_len,
        random_crop=True,
        limit=limit,
        shuffle_files=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_unsup,
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    # enc = Encoder(K=args.n_src).to(device)
    # dec = Decoder(K=args.n_src).to(device)
    F = args.n_fft // 2 + 1
    M = len(select_channels) if select_channels is not None else 6
    enc = Encoder(F=F, M=M, K=args.n_src).to(device)
    dec = Decoder(F=F, K=args.n_src).to(device)
    optim = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr)

    def save_ckpt(path: str, epoch: int):
        torch.save({"epoch": epoch, "model": {"enc": enc.state_dict(), "dec": dec.state_dict()}, "args": vars(args)}, path)

    history = []
    last_path = os.path.join(args.exp_dir, "last.pt")

    for ep in range(1, args.epochs + 1):
        # beta = cyclic_beta(ep, args.kl_cycle, args.kl_ratio, args.kl_max_beta)
        max_beta = args.kl_max_beta_first if ep <= args.kl_first_epochs else args.kl_max_beta
        beta = cyclic_beta(ep, args.kl_cycle, args.kl_ratio, max_beta)

        enc.train()
        dec.train()
        window = torch.hann_window(args.n_fft, device=device)

        losses = []
        t0 = time.time()
        for step, (mix_btc, ilens, _utt_ids) in enumerate(train_loader, start=1):
            mix_btc = mix_btc.to(device)                # (B,T,C)
            ilens = ilens.to(device)
            mix_btc = mix_btc[:, : int(ilens.max().item()), :]  # truncate padding BEFORE STFT
            x = stft_mc(mix_btc, args.n_fft, args.hop, window)  # (B,T',F,M)

            # scale normalization
            scale = torch.sqrt((x.abs().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-12))
            x = x / scale

            B, TT, F, M = x.shape
            norm_tf = float(TT * F)
            xx = batch_to_xx(x)

            q = enc(x, distribution=True)               # Normal
            z = q.rsample()
            lm = dec(z).clamp_min(1e-10)                # (B,TT,F,K)

            # KL(q||N(0,I))
            mu, std = q.loc, q.scale
            kl = 0.5 * (mu.pow(2) + std.pow(2) - 2.0 * torch.log(std + 1e-12) - 1.0)
            # kl = kl.sum(dim=list(range(1, kl.dim()))).mean()
            kl = kl.mean()

            # per-sample H updates (baseline, simplest & stable)
            nlls = []
            for b in range(B):
                H = init_H(F=F, K=args.n_src, M=M, device=device)
                for _ in range(int(args.n_hiter)):
                    H = update_H_em(xx[b], lm[b], H)
                nlls.append(nll_gaussian(xx[b], lm[b], H))
            # nll = torch.stack(nlls).mean()
            num_elements = B * TT * F
            nll = torch.stack(nlls).sum() / num_elements

            loss = nll + float(beta) * kl

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 5.0)
            optim.step()

            # losses.append(float(loss.item()))
            losses.append(float(loss.item()))
            if step % args.log_interval == 0 or step == 1:
                it_s = step / max(time.time() - t0, 1e-9)
                print(f"  [batch {step:05d}/{len(train_loader)}] loss={loss.item():.6f} beta={beta:.3f} ({it_s:.2f} it/s)")

        tr = float(np.mean(losses)) if losses else float("nan")
        print(f"[E{ep:03d}] train={tr:.4f}")
        history.append({"epoch": ep, "train_loss": tr, "beta": beta})

        if ep % 5 == 0:
            save_ckpt(os.path.join(args.exp_dir, f"epoch{ep:03d}.pt"), ep)
        save_ckpt(last_path, ep)

    # write history
    with open(os.path.join(args.exp_dir, "loss_history.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "beta"])
        w.writeheader()
        w.writerows(history)

    # plot loss curve
    epochs = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    betas = [h["beta"] for h in history]

    plt.figure()
    plt.plot(epochs, losses, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss (NLL + beta*KL)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.exp_dir, "loss_curve.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # plot beta curve
    plt.figure()
    plt.plot(epochs, betas, label="beta")
    plt.xlabel("epoch")
    plt.ylabel("KL weight beta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_png2 = os.path.join(args.exp_dir, "beta_curve.png")
    plt.savefig(out_png2, dpi=150)
    plt.close()

    print(f"[DONE] wrote: {out_png}")
    print(f"[DONE] wrote: {out_png2}")
    print(f"[DONE] last ckpt: {last_path}")


if __name__ == "__main__":
    main()