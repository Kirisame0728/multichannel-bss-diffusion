# metrics/sisdr.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .core import compute_pairwise, pit_from_pairwise_scores, PITResult, _to_3d_torch, _crop_to_min_time, _zero_mean


def _pairwise_sisdr_torch(
    est_1d: torch.Tensor,
    ref_1d: torch.Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> torch.Tensor:
    """
    SI-SDR between two 1D signals (T,).
    Higher is better.
    """
    if est_1d.dim() != 1 or ref_1d.dim() != 1:
        raise ValueError(f"Expected 1D tensors. Got est={tuple(est_1d.shape)}, ref={tuple(ref_1d.shape)}")

    if est_1d.shape[0] != ref_1d.shape[0]:
        T = min(est_1d.shape[0], ref_1d.shape[0])
        est_1d = est_1d[:T]
        ref_1d = ref_1d[:T]

    if zero_mean:
        est_1d = est_1d - est_1d.mean()
        ref_1d = ref_1d - ref_1d.mean()

    # projection of est onto ref
    ref_energy = torch.sum(ref_1d * ref_1d) + eps
    scale = torch.sum(est_1d * ref_1d) / ref_energy
    s_target = scale * ref_1d
    e_noise = est_1d - s_target

    target_energy = torch.sum(s_target * s_target) + eps
    noise_energy = torch.sum(e_noise * e_noise) + eps

    sisdr = 10.0 * torch.log10(target_energy / noise_energy)
    return sisdr


def sisdr_pairwise_matrix(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> torch.Tensor:
    """
    Returns pairwise SI-SDR matrix of shape (B, N, N) on CPU.
    """
    def fn(e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return _pairwise_sisdr_torch(e, r, eps=eps, zero_mean=zero_mean)

    return compute_pairwise(est, ref, pairwise_fn=fn, device=device)


def sisdr_pit(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tuple[float, PITResult]:
    """
    Compute PIT SI-SDR.
    Returns:
      score_avg: float (mean over sources, then mean over batch)
      pit_result: PITResult for single-sample case; for batch, it's representative
                  (use sisdr_pit_batch if you need per-sample perms).
    """
    pairwise = sisdr_pairwise_matrix(est, ref, device=device, eps=eps, zero_mean=zero_mean)  # (B,N,N) cpu
    res = pit_from_pairwise_scores(pairwise)
    return res.best_score, res


def sisdr_pit_batch(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> Tuple[np.ndarray, list]:
    """
    Batch-friendly PIT SI-SDR.
    Returns:
      scores: (B,) best avg SI-SDR per sample
      results: list[PITResult] length B with per-sample best_perm and per-source scores
    """
    pairwise = sisdr_pairwise_matrix(est, ref, device=device, eps=eps, zero_mean=zero_mean)  # (B,N,N) cpu
    pw = pairwise.numpy()
    B, N, _ = pw.shape
    results = []
    scores = np.zeros((B,), dtype=np.float64)

    # brute force per sample (B is usually small in eval)
    from .core import _enumerate_perms
    perms = _enumerate_perms(N)

    for b in range(B):
        best_score = -np.inf
        best_perm = perms[0]
        for perm in perms:
            diag = pw[b, np.arange(N), np.array(perm)]
            avg = float(diag.mean())
            if avg > best_score:
                best_score = avg
                best_perm = perm
                best_diag = diag
        results.append(PITResult(best_perm=best_perm, best_score=best_score, per_source_scores=best_diag.astype(np.float64)))
        scores[b] = best_score

    return scores, results
