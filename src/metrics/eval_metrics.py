# eval_metrics.py

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch


def _pairwise_sisdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    est = est - est.mean()
    ref = ref - ref.mean()

    ref_pow = torch.sum(ref * ref) + eps
    scale = torch.sum(ref * est) / ref_pow

    s_true = scale * ref
    e_res = est - s_true

    true_pow = torch.sum(s_true * s_true) + eps
    res_pow = torch.sum(e_res * e_res) + eps
    return 10.0 * torch.log10(true_pow / res_pow)


def sisdr_batch(
    est: torch.Tensor,
    ref: torch.Tensor,
    max_samples: Optional[int] = None,
    eps: float = 1e-8,
) -> float:
    if est.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"Expected 2D tensors (n_src, T). Got est={tuple(est.shape)}, ref={tuple(ref.shape)}")
    if est.shape[0] != ref.shape[0]:
        raise ValueError(f"n_src mismatch: est={tuple(est.shape)}, ref={tuple(ref.shape)}")

    if max_samples is not None:
        est = est[:, :max_samples]
        ref = ref[:, :max_samples]

    n_src = int(est.shape[0])
    T = min(int(est.shape[1]), int(ref.shape[1]))
    est = est[:, :T]
    ref = ref[:, :T]

    if n_src == 1:
        return float(_pairwise_sisdr(est[0], ref[0], eps=eps).item())

    pair = torch.empty((n_src, n_src), dtype=torch.float32, device=est.device)
    for i in range(n_src):
        for j in range(n_src):
            pair[i, j] = _pairwise_sisdr(est[i], ref[j], eps=eps)

    best = None
    for perm in itertools.permutations(range(n_src)):
        s = 0.0
        for i, j in enumerate(perm):
            s += float(pair[i, j].item())
        s /= n_src
        if best is None or s > best:
            best = s

    return float(best)


def pesq_batch(est: np.ndarray, ref: np.ndarray, sr: int = 8000) -> float:
    try:
        from pesq import pesq as pesq_fn
    except Exception as e:
        raise RuntimeError("Missing dependency 'pesq'. Install it with: pip install pesq") from e

    if est.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"Expected (n_src, T) arrays. Got est={est.shape}, ref={ref.shape}")
    if est.shape[0] != ref.shape[0]:
        raise ValueError(f"n_src mismatch: est={est.shape}, ref={ref.shape}")

    mode = "nb" if sr == 8000 else "wb"

    n_src = int(est.shape[0])
    T = min(int(est.shape[1]), int(ref.shape[1]))
    est = est[:, :T]
    ref = ref[:, :T]

    if n_src == 1:
        return float(pesq_fn(sr, ref[0], est[0], mode))

    pair = np.zeros((n_src, n_src), dtype=np.float64)
    for i in range(n_src):
        for j in range(n_src):
            pair[i, j] = float(pesq_fn(sr, ref[j], est[i], mode))

    best = None
    for perm in itertools.permutations(range(n_src)):
        s = 0.0
        for i, j in enumerate(perm):
            s += pair[i, j]
        s /= n_src
        if best is None or s > best:
            best = s

    return float(best)


def estoi_batch(est: np.ndarray, ref: np.ndarray, sr: int = 8000) -> float:
    try:
        from pystoi import stoi as stoi_fn
    except Exception as e:
        raise RuntimeError("Missing dependency 'pystoi'. Install it with: pip install pystoi") from e

    if est.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"Expected (n_src, T) arrays. Got est={est.shape}, ref={ref.shape}")
    if est.shape[0] != ref.shape[0]:
        raise ValueError(f"n_src mismatch: est={est.shape}, ref={ref.shape}")

    n_src = int(est.shape[0])
    T = min(int(est.shape[1]), int(ref.shape[1]))
    est = est[:, :T]
    ref = ref[:, :T]

    if n_src == 1:
        return float(stoi_fn(ref[0], est[0], sr, extended=True))

    pair = np.zeros((n_src, n_src), dtype=np.float64)
    for i in range(n_src):
        for j in range(n_src):
            pair[i, j] = float(stoi_fn(ref[j], est[i], sr, extended=True))

    best = None
    for perm in itertools.permutations(range(n_src)):
        s = 0.0
        for i, j in enumerate(perm):
            s += pair[i, j]
        s /= n_src
        if best is None or s > best:
            best = s

    return float(best)
