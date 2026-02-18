# metrics/perceptual.py
from __future__ import annotations

from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from .core import PITResult, pit_from_pairwise_scores, _to_3d_torch, _crop_to_min_time


def _import_pesq():
    try:
        from pesq import pesq  # type: ignore
        return pesq
    except Exception as e:
        raise ImportError("Missing dependency 'pesq'. Install with: pip install pesq") from e


def _import_pystoi():
    try:
        from pystoi import stoi  # type: ignore
        return stoi
    except Exception as e:
        raise ImportError("Missing dependency 'pystoi'. Install with: pip install pystoi") from e


def _to_numpy_1d(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got {x.shape}")
    return x.astype(np.float32, copy=False)


def _pairwise_pesq(
    est_1d: Union[np.ndarray, torch.Tensor],
    ref_1d: Union[np.ndarray, torch.Tensor],
    sr: int,
    mode: str = "wb",
) -> float:
    """
    PESQ score. Higher is better.
    mode: "wb" for 16k, "nb" for 8k typically.
    """
    pesq = _import_pesq()
    e = _to_numpy_1d(est_1d)
    r = _to_numpy_1d(ref_1d)
    T = min(len(e), len(r))
    e, r = e[:T], r[:T]

    # pesq expects python floats, returns float
    return float(pesq(sr, r, e, mode))


def _pairwise_estoi(
    est_1d: Union[np.ndarray, torch.Tensor],
    ref_1d: Union[np.ndarray, torch.Tensor],
    sr: int,
    extended: bool = True,
) -> float:
    """
    eSTOI/STOI. Higher is better.
    extended=True -> eSTOI (if pystoi supports it); pystoi uses "extended" flag.
    """
    stoi = _import_pystoi()
    e = _to_numpy_1d(est_1d)
    r = _to_numpy_1d(ref_1d)
    T = min(len(e), len(r))
    e, r = e[:T], r[:T]
    return float(stoi(r, e, sr, extended=extended))


def _pairwise_matrix_numpy_metric(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    pairwise_fn,
    sr: int,
) -> np.ndarray:
    """
    Compute pairwise matrix (B,N,N) for a numpy-based metric.
    """
    est_t = _to_3d_torch(est, device=None)
    ref_t = _to_3d_torch(ref, device=None)
    est_t, ref_t = _crop_to_min_time(est_t, ref_t)

    if est_t.shape[:2] != ref_t.shape[:2]:
        raise ValueError(f"est/ref must match on (B,N). Got est={tuple(est_t.shape)}, ref={tuple(ref_t.shape)}")

    B, N, _ = est_t.shape
    pw = np.zeros((B, N, N), dtype=np.float64)

    for b in range(B):
        for i in range(N):      # ref index
            for j in range(N):  # est index
                pw[b, i, j] = pairwise_fn(est_t[b, j], ref_t[b, i], sr)

    return pw


def pesq_pit(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    sr: int,
    mode: str = "wb",
) -> Tuple[float, PITResult]:
    pw = _pairwise_matrix_numpy_metric(
        est, ref, pairwise_fn=lambda e, r, sr_: _pairwise_pesq(e, r, sr_, mode=mode), sr=sr
    )  # (B,N,N)
    res = pit_from_pairwise_scores(pw)
    return res.best_score, res


def estoi_pit(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    sr: int,
    extended: bool = True,
) -> Tuple[float, PITResult]:
    pw = _pairwise_matrix_numpy_metric(
        est, ref, pairwise_fn=lambda e, r, sr_: _pairwise_estoi(e, r, sr_, extended=extended), sr=sr
    )
    res = pit_from_pairwise_scores(pw)
    return res.best_score, res
