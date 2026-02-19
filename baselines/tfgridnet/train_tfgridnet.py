#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import argparse
import time
from typing import List, Optional, Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def _add_repo_root_to_syspath() -> str:
    """
    Add repository root to sys.path.

    Purpose:
        Enable `src.*` imports when this script is executed from `baselines/tfgridnet/`.

    Returns:
        Absolute path of repository root.
    """
    here = os.path.abspath(os.path.dirname(__file__))            # .../baselines/tfgridnet
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))  # repo root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_REPO_ROOT = _add_repo_root_to_syspath()

# Repository modules
from src.dataloaders.dataset_mix2_train import Mix2TrainDataset  # type: ignore
from tfgridnet_separator import TFGridNet


def _parse_int_list(s: str) -> Optional[List[int]]:
    """
    Parse comma-separated integer list.

    Examples:
        "0,2,4" -> [0, 2, 4]
        ""      -> None

    Args:
        s: Comma-separated integers.

    Returns:
        List of integers or None if input is empty.
    """
    s = (s or "").strip()
    if s == "":
        return None
    return [int(x) for x in s.split(",") if x.strip() != ""]


class Mix2TrainAdapter(torch.utils.data.Dataset):
    """
    Dataset adapter for training script compatibility.

    This adapter normalizes the sample format produced by `Mix2TrainDataset`
    into the format expected by the existing training loop.

    Per-sample output (before collate):
        mix_tc:    (T, C) float32
        tgt_kct:   (K, C, T) float32
        ilen:      int, length T before padding
        utt_id:    str
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        n_src: int,
        sample_rate: int,
        num_channels: int,
        select_channels: Optional[List[int]],
        target_mode: str,
        max_len: Optional[int],
        random_crop: bool,
        limit: Optional[int],
        shuffle_files: bool,
    ):
        self.ds = Mix2TrainDataset(
            root_dir=root_dir,
            split=split,
            sample_rate=sample_rate,
            n_src=n_src,
            num_channels=num_channels,
            select_channels=select_channels,
            target_mode=target_mode,
            max_len=max_len,
            random_crop=random_crop,
            limit=limit,
            shuffle_files=shuffle_files,
        )
        self.n_src = n_src

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        """
        Load one sample and normalize into (mix_tc, tgt_kct, ilen, utt_id).

        Notes:
            - Mixture is expected to be shaped (C, T) from the dataset.
            - Target sources are expected to be shaped (K, C, T) or (K, T).
        """
        item = self.ds[idx]

        if isinstance(item, dict):
            utt_id = str(item.get("utt_id", f"sample_{idx}"))
            mixture = item["mixture"]  # expected (C, T)

            if "sources" in item and item["sources"] is not None:
                sources = item["sources"]  # expected (K, C, T) or (K, T)
            else:
                early = item.get("early", None)
                tail = item.get("tail", None)
                if early is None or tail is None:
                    raise KeyError("Dataset item must contain 'sources' or both 'early' and 'tail'.")
                sources = early + tail  # (K, C, T)
        else:
            # Tuple-return datasets are supported for robustness.
            if len(item) == 4:
                mixture, early, tail, utt_id = item
                sources = early + tail
            elif len(item) == 3:
                mixture, sources, utt_id = item
            else:
                raise ValueError("Unsupported dataset item format.")
            utt_id = str(utt_id)

        if not torch.is_tensor(mixture):
            mixture = torch.as_tensor(mixture)
        mixture = mixture.float()

        if not torch.is_tensor(sources):
            sources = torch.as_tensor(sources)
        sources = sources.float()

        # (C, T) -> (T, C)
        mix_tc = mixture.transpose(0, 1).contiguous()

        # Normalize sources to (K, C, T)
        if sources.dim() == 3:
            tgt_kct = sources.contiguous()
        elif sources.dim() == 2:
            tgt_kct = sources[:, None, :].contiguous()
        else:
            raise ValueError(f"Unexpected sources shape: {tuple(sources.shape)}")

        ilen = int(mix_tc.shape[0])
        return mix_tc, tgt_kct, ilen, utt_id


def pad_collate_mix2(batch):
    """
    Pad-and-stack collate function.

    Purpose:
        Convert a list of variable-length samples into fixed-size batch tensors
        by zero-padding to the maximum length within the batch.

    Input items:
        mix_tc:  (T, C)
        tgt_kct: (K, C, T)
        ilen:    int
        utt_id:  str

    Returns:
        mix:      (B, T_max, C)
        tgt_full: (B, K, C, T_max)
        ilens:    (B,)
        utt_ids:  list[str]
    """
    mixes, tgts, ilens, utt_ids = zip(*batch)
    B = len(mixes)
    T_max = max(int(x.shape[0]) for x in mixes)

    C = int(mixes[0].shape[1])
    mix_out = mixes[0].new_zeros((B, T_max, C))
    for b in range(B):
        t = int(mixes[b].shape[0])
        mix_out[b, :t, :] = mixes[b]

    K = int(tgts[0].shape[0])
    C_tgt = int(tgts[0].shape[1])
    tgt_out = tgts[0].new_zeros((B, K, C_tgt, T_max))
    for b in range(B):
        t = int(tgts[b].shape[-1])
        tgt_out[b, :, :, :t] = tgts[b]

    ilens_out = torch.tensor(ilens, dtype=torch.long)
    return mix_out, tgt_out, ilens_out, list(utt_ids)


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute SI-SDR per batch element.

    Args:
        est: (B, T) estimated signal
        ref: (B, T) reference signal
        eps: numerical epsilon

    Returns:
        (B,) SI-SDR values in dB.
    """
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    ref_energy = torch.sum(ref**2, dim=-1, keepdim=True) + eps
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj

    ratio = (torch.sum(proj**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def pit_si_sdr_loss(est_list: List[torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
    """
    PIT SI-SDR loss (negative best-permutation SI-SDR).

    Args:
        est_list: list length K, each tensor shaped (B, T)
        tgt: (B, K, T)

    Returns:
        Scalar loss.
    """
    K = len(est_list)
    assert tgt.dim() == 3 and tgt.size(1) == K

    pair = []
    for i in range(K):
        row = []
        for j in range(K):
            row.append(si_sdr(est_list[i], tgt[:, j, :]))  # (B,)
        pair.append(torch.stack(row, dim=1))  # (B, K)
    pair = torch.stack(pair, dim=1)  # (B, K, K)

    if K == 2:
        s1 = pair[:, 0, 0] + pair[:, 1, 1]
        s2 = pair[:, 0, 1] + pair[:, 1, 0]
        best = torch.maximum(s1, s2)
        return -best.mean()

    if K == 3:
        perms = [
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        ]
        scores = []
        for p in perms:
            scores.append(pair[:, 0, p[0]] + pair[:, 1, p[1]] + pair[:, 2, p[2]])
        best = torch.stack(scores, dim=1).max(dim=1).values
        return -best.mean()

    raise NotImplementedError("PIT for K>3 is not implemented in this trainer.")


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    ref_mic: int,
    grad_clip: float,
    log_interval: int = 20,
) -> float:
    """
    Run one training epoch.

    Args:
        model: separation model
        loader: DataLoader
        optimizer: optimizer instance
        scaler: GradScaler (AMP)
        device: torch device
        ref_mic: reference microphone index used for supervision
        grad_clip: gradient clipping norm (<=0 disables)
        log_interval: print interval in steps

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    losses = []

    total_steps = len(loader)
    start_time = time.time()
    recent = []

    for step, (mix, tgt_full, ilens, _utt_ids) in enumerate(loader, start=1):
        mix = mix.to(device)              # (B, T, C)
        ilens = ilens.to(device)          # (B,)
        tgt_full = tgt_full.to(device)    # (B, K, C, T)

        tgt = tgt_full[:, :, ref_mic, :]  # (B, K, T)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            est_list, _, _ = model(mix, ilens)  # list length K, each (B, T)
            loss = pit_si_sdr_loss(est_list, tgt)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        loss_item = float(loss.item())
        losses.append(loss_item)

        recent.append(loss_item)
        if len(recent) > log_interval:
            recent.pop(0)

        if (step % log_interval == 0) or (step == 1) or (step == total_steps):
            elapsed = time.time() - start_time
            it_s = step / max(elapsed, 1e-9)
            eta_s = (total_steps - step) / max(it_s, 1e-9)
            avg = sum(recent) / max(len(recent), 1)
            print(
                f"  [batch {step:05d}/{total_steps}] "
                f"loss={loss_item:.4f} (avg{len(recent)}={avg:.4f}) "
                f"({it_s:.2f} it/s, eta {eta_s/60:.1f} min)"
            )

    return float(np.mean(losses)) if losses else float("nan")


def save_csv(path: str, history: List[dict]) -> None:
    """Write training history as a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        for row in history:
            w.writerow(row)


def plot_png(path: str, history: List[dict]) -> None:
    """Plot training curve and write as PNG."""
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]

    plt.figure()
    plt.plot(epochs, tr, label="train")
    plt.xlabel("epoch")
    plt.ylabel("loss = -PIT SI-SDR")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser("TF-GridNet trainer")

    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--exp_dir", type=str, required=True)

    p.add_argument("--sample_rate", type=int, default=8000)
    p.add_argument("--n_src", type=int, default=2)
    p.add_argument("--target_mode", type=str, default="early+late", choices=["early", "early+late"])

    p.add_argument("--select_channels", type=str, default="",
                   help='Comma-separated channel indices, e.g. "0,2,4". Empty uses all channels.')
    p.add_argument("--ref_mic", type=int, default=0)

    p.add_argument("--max_len", type=int, default=32000, help="Crop length in samples (<=0 disables).")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=5.0)

    p.add_argument("--log_interval", type=int, default=20, help="Print loss every N batches.")

    p.add_argument("--n_fft", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)

    args = p.parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)

    select_channels = _parse_int_list(args.select_channels)
    max_len = None if args.max_len <= 0 else args.max_len

    train_set = Mix2TrainAdapter(
        root_dir=args.root_dir,
        split=args.train_split,
        n_src=args.n_src,
        target_mode=args.target_mode,
        sample_rate=args.sample_rate,
        num_channels=(len(select_channels) if select_channels is not None else 6),
        select_channels=select_channels,
        max_len=max_len,
        random_crop=True,
        limit=1000,
        shuffle_files=True,
    )

    if len(train_set) == 0:
        raise RuntimeError("Training dataset is empty.")

    if select_channels is None:
        mix0, _tgt0, _ilen0, _uid0 = train_set[0]
        n_imics = int(mix0.shape[1])
    else:
        n_imics = len(select_channels)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_mix2,
        persistent_workers=(args.num_workers > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    model = TFGridNet(
        input_dim=0,
        n_srcs=args.n_src,
        n_fft=args.n_fft,
        stride=args.stride,
        window="hann",
        n_imics=n_imics,
        n_layers=6,
        lstm_hidden_units=192,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=48,
        emb_ks=4,
        emb_hs=1,
        activation="prelu",
        eps=1.0e-5,
        use_builtin_complex=False,
        ref_channel=-1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    history = []
    last_path = os.path.join(args.exp_dir, "last.pt")

    for ep in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            ref_mic=args.ref_mic,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
        )
        print(f"[E{ep:03d}] train={tr:.4f}")
        history.append({"epoch": ep, "train_loss": tr})

        if ep % 5 == 0:
            ck = os.path.join(args.exp_dir, f"epoch{ep:03d}.pt")
            torch.save({"epoch": ep, "model": model.state_dict(), "args": vars(args)}, ck)

        torch.save(
            {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            },
            last_path,
        )

    csv_path = os.path.join(args.exp_dir, "loss_history.csv")
    png_path = os.path.join(args.exp_dir, "loss_curve.png")
    save_csv(csv_path, history)
    plot_png(png_path, history)

    print(f"[DONE] wrote: {csv_path}")
    print(f"[DONE] wrote: {png_path}")
    print(f"[DONE] last ckpt: {last_path}")


if __name__ == "__main__":
    main()
