# metrics/bss_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .core import PITResult, _to_3d_torch, _crop_to_min_time, _enumerate_perms


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class BSSEvalScores:
    """Averaged scores (across sources; across batch if provided)."""
    SDR: float
    SIR: float
    SAR: float


def _to_numpy_3d(x: ArrayLike) -> np.ndarray:
    """(N,T) or (B,N,T) -> (B,N,T) float64 numpy"""
    x_t = _to_3d_torch(x, device=None)
    return x_t.detach().cpu().numpy().astype(np.float64, copy=False)


def _zero_mean_np(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=-1, keepdims=True)


def _proj(u: np.ndarray, v: np.ndarray, eps: float) -> np.ndarray:
    """
    Project vector u onto vector v: <u,v>/<v,v> * v
    u,v: (T,)
    """
    denom = np.dot(v, v) + eps
    return (np.dot(u, v) / denom) * v


def _bss_eval_one_pair(
    est: np.ndarray,
    ref_all: np.ndarray,
    target_index: int,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    BSS_EVAL-like decomposition for one estimated source against all reference sources.

    est: (T,)
    ref_all: (N,T) reference sources
    target_index: which ref source is considered the 'target' for SDR/SIR/SAR

    Returns (SDR, SIR, SAR) in dB.
    """
    # Target component: projection of est onto the target reference
    s_target = _proj(est, ref_all[target_index], eps)

    # Interference: sum of projections onto the other references
    e_interf = np.zeros_like(est)
    for k in range(ref_all.shape[0]):
        if k == target_index:
            continue
        e_interf += _proj(est, ref_all[k], eps)

    # Artifact: residual
    e_artif = est - (s_target + e_interf)

    # Energies
    target_energy = np.sum(s_target ** 2) + eps
    interf_energy = np.sum(e_interf ** 2) + eps
    artif_energy = np.sum(e_artif ** 2) + eps
    total_err_energy = np.sum((e_interf + e_artif) ** 2) + eps

    SDR = 10.0 * np.log10(target_energy / total_err_energy)
    SIR = 10.0 * np.log10(target_energy / interf_energy)
    SAR = 10.0 * np.log10((target_energy + interf_energy) / artif_energy)
    return float(SDR), float(SIR), float(SAR)


def bss_eval_pairwise_matrices(
    est: ArrayLike,
    ref: ArrayLike,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise (B,N,N) matrices for SDR/SIR/SAR.

    Return:
      pw_sdr, pw_sir, pw_sar: np.ndarray float64 with shape (B,N,N)
      where entry [b,i,j] is score when ref source i is target and est source j is estimated.
    """
    est_np = _to_numpy_3d(est)  # (B,N,T)
    ref_np = _to_numpy_3d(ref)  # (B,N,T)
    est_np, ref_np = _crop_to_min_time(torch.from_numpy(est_np), torch.from_numpy(ref_np))
    est_np = est_np.numpy()
    ref_np = ref_np.numpy()

    if est_np.shape[:2] != ref_np.shape[:2]:
        raise ValueError(f"est/ref must match on (B,N). Got est={est_np.shape}, ref={ref_np.shape}")

    if zero_mean:
        est_np = _zero_mean_np(est_np)
        ref_np = _zero_mean_np(ref_np)

    B, N, T = est_np.shape
    pw_sdr = np.zeros((B, N, N), dtype=np.float64)
    pw_sir = np.zeros((B, N, N), dtype=np.float64)
    pw_sar = np.zeros((B, N, N), dtype=np.float64)

    for b in range(B):
        ref_all = ref_np[b]  # (N,T)
        for i in range(N):      # target ref index
            for j in range(N):  # est index
                sdr, sir, sar = _bss_eval_one_pair(est_np[b, j], ref_all, target_index=i, eps=eps)
                pw_sdr[b, i, j] = sdr
                pw_sir[b, i, j] = sir
                pw_sar[b, i, j] = sar

    return pw_sdr, pw_sir, pw_sar


def bss_eval_pit(
    est: ArrayLike,
    ref: ArrayLike,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tuple[BSSEvalScores, PITResult]:
    """
    PIT BSS_EVAL metrics.
    - Build pairwise SDR/SIR/SAR matrices
    - Choose permutation that maximizes average SDR
    - Return averaged SDR/SIR/SAR after applying that perm

    Returns:
      scores: BSSEvalScores averaged over sources and batch
      pit: PITResult (for batch, best_perm is representative; per_source_scores is from sample 0)
    """
    pw_sdr, pw_sir, pw_sar = bss_eval_pairwise_matrices(est, ref, eps=eps, zero_mean=zero_mean)
    B, N, _ = pw_sdr.shape
    perms = _enumerate_perms(N)

    best_scores = np.full((B,), -np.inf, dtype=np.float64)
    best_perms: List[Tuple[int, ...]] = [perms[0]] * B

    for perm in perms:
        diag = pw_sdr[:, np.arange(N), np.array(perm)]  # (B,N)
        avg = diag.mean(axis=1)                         # (B,)
        better = avg > best_scores
        if np.any(better):
            best_scores[better] = avg[better]
            for b in np.where(better)[0]:
                best_perms[b] = perm

    # Aggregate final metrics under per-sample best perm
    sdr_list = []
    sir_list = []
    sar_list = []
    for b in range(B):
        perm = best_perms[b]
        sdr_list.append(pw_sdr[b, np.arange(N), np.array(perm)].mean())
        sir_list.append(pw_sir[b, np.arange(N), np.array(perm)].mean())
        sar_list.append(pw_sar[b, np.arange(N), np.array(perm)].mean())

    scores = BSSEvalScores(
        SDR=float(np.mean(sdr_list)),
        SIR=float(np.mean(sir_list)),
        SAR=float(np.mean(sar_list)),
    )

    # Build PITResult (representative for batch)
    rep_perm = best_perms[0]
    rep_diag = pw_sdr[0, np.arange(N), np.array(rep_perm)]
    pit = PITResult(best_perm=rep_perm, best_score=float(np.mean(best_scores)), per_source_scores=rep_diag.astype(np.float64))
    return scores, pit


def bss_eval_no_pit(
    est: ArrayLike,
    ref: ArrayLike,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> BSSEvalScores:
    """
    No-PIT BSS_EVAL metrics, assumes aligned ordering: est[i] matches ref[i].
    Returns averaged SDR/SIR/SAR over sources and batch.
    """
    pw_sdr, pw_sir, pw_sar = bss_eval_pairwise_matrices(est, ref, eps=eps, zero_mean=zero_mean)
    B, N, _ = pw_sdr.shape
    diag_sdr = pw_sdr[:, np.arange(N), np.arange(N)].mean()
    diag_sir = pw_sir[:, np.arange(N), np.arange(N)].mean()
    diag_sar = pw_sar[:, np.arange(N), np.arange(N)].mean()
    return BSSEvalScores(SDR=float(diag_sdr), SIR=float(diag_sir), SAR=float(diag_sar))
