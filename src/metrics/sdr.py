# metrics/sdr.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch

from .core import compute_pairwise, pit_from_pairwise_scores, PITResult


def _pairwise_sdr_torch(
    est_1d: torch.Tensor,
    ref_1d: torch.Tensor,
    eps: float = 1e-8,
    mode: str = "plain",
    zero_mean: bool = False,
) -> torch.Tensor:
    """
    SDR between two 1D signals (T,).

    mode:
      - "plain": 10log10( ||ref||^2 / ||ref - est||^2 )
      - "proj" : 10log10( ||alpha*ref||^2 / ||alpha*ref - est||^2 ), alpha = <est,ref>/<ref,ref>
                (scale-optimized, but if zero_mean=False it's not SI-SDR; if zero_mean=True it becomes SI-SDR-like)
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

    if mode not in ("plain", "proj"):
        raise ValueError("mode must be 'plain' or 'proj'")

    if mode == "plain":
        num = torch.sum(ref_1d * ref_1d) + eps
        den = torch.sum((ref_1d - est_1d) ** 2) + eps
        return 10.0 * torch.log10(num / den)

    # mode == "proj"
    ref_energy = torch.sum(ref_1d * ref_1d) + eps
    alpha = torch.sum(est_1d * ref_1d) / ref_energy
    ref_scaled = alpha * ref_1d
    num = torch.sum(ref_scaled * ref_scaled) + eps
    den = torch.sum((ref_scaled - est_1d) ** 2) + eps
    return 10.0 * torch.log10(num / den)


def sdr_pairwise_matrix(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    mode: str = "plain",
    zero_mean: bool = False,
) -> torch.Tensor:
    """
    Returns pairwise SDR matrix of shape (B, N, N) on CPU.
    """
    def fn(e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return _pairwise_sdr_torch(e, r, eps=eps, mode=mode, zero_mean=zero_mean)

    return compute_pairwise(est, ref, pairwise_fn=fn, device=device)


def sdr_pit(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    mode: str = "plain",
    zero_mean: bool = False,
) -> Tuple[float, PITResult]:
    """
    Compute PIT SDR.
    Returns:
      score_avg: float (mean over sources; if batch, averaged over batch in a representative manner)
      pit_result: PITResult (representative for batch)
    """
    pairwise = sdr_pairwise_matrix(est, ref, device=device, eps=eps, mode=mode, zero_mean=zero_mean)  # (B,N,N) cpu
    res = pit_from_pairwise_scores(pairwise)
    return res.best_score, res


def sdr_no_pit(
    est: Union[np.ndarray, torch.Tensor],
    ref: Union[np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    eps: float = 1e-8,
    mode: str = "plain",
    zero_mean: bool = False,
) -> float:
    """
    SDR without PIT, assuming already aligned sources (est[i] matches ref[i]).
    Computes mean over sources (and batch if provided).
    """
    pairwise = sdr_pairwise_matrix(est, ref, device=device, eps=eps, mode=mode, zero_mean=zero_mean)  # (B,N,N)
    pw = pairwise.numpy()  # CPU
    B, N, _ = pw.shape
    diag = pw[:, np.arange(N), np.arange(N)]  # (B,N)
    return float(diag.mean())
