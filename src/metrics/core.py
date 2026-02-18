from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, Dict, Any

import itertools
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass(frozen=True)
class PITResult:
    """Result of PIT assignment."""
    best_perm: Tuple[int, ...]          # est index for each ref index, length = n_src
    best_score: float                   # scalar: best average score across sources
    per_source_scores: np.ndarray       # shape (n_src,), scores after applying best_perm


def _is_torch(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _to_3d_torch(x: ArrayLike, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Normalize input to torch.Tensor with shape (B, N, T).
    Accepts:
      - (N, T)
      - (B, N, T)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")

    if x.dim() == 2:
        x = x.unsqueeze(0)  # (1, N, T)
    elif x.dim() != 3:
        raise ValueError(f"Expected shape (N,T) or (B,N,T), got {tuple(x.shape)}")

    if device is not None:
        x = x.to(device)
    return x


def _check_compat(est: torch.Tensor, ref: torch.Tensor) -> Tuple[int, int, int]:
    """
    est/ref: (B, N, T) torch tensors
    Returns (B, N, T_min) after time alignment check.
    """
    if est.shape[:2] != ref.shape[:2]:
        raise ValueError(f"est/ref must match on (B,N). Got est={tuple(est.shape)}, ref={tuple(ref.shape)}")
    B, N, T_est = est.shape
    _, _, T_ref = ref.shape
    T = min(T_est, T_ref)
    if T_est != T_ref:
        # Crop to common length to avoid silent broadcasting bugs.
        est = est[..., :T]
        ref = ref[..., :T]
    return B, N, T


def _crop_to_min_time(est: torch.Tensor, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if est.shape[-1] == ref.shape[-1]:
        return est, ref
    T = min(est.shape[-1], ref.shape[-1])
    return est[..., :T], ref[..., :T]


def _zero_mean(x: torch.Tensor, dim: int = -1, eps: float = 0.0) -> torch.Tensor:
    # eps kept for API symmetry; not used
    return x - x.mean(dim=dim, keepdim=True)


def _enumerate_perms(n_src: int) -> List[Tuple[int, ...]]:
    return list(itertools.permutations(range(n_src)))


def pit_from_pairwise_scores(pairwise: ArrayLike) -> PITResult:
    """
    Solve PIT given a pairwise score matrix per sample.

    pairwise can be:
      - (N, N)  : single sample
      - (B, N, N): batch
    The score is assumed "higher is better".
    We maximize the mean diagonal score under permutation.
    """
    if isinstance(pairwise, torch.Tensor):
        pw = pairwise.detach().cpu().numpy()
    elif isinstance(pairwise, np.ndarray):
        pw = pairwise
    else:
        raise TypeError(f"pairwise must be torch.Tensor or np.ndarray, got {type(pairwise)}")

    if pw.ndim == 2:
        pw = pw[None, ...]  # (1,N,N)
    if pw.ndim != 3 or pw.shape[1] != pw.shape[2]:
        raise ValueError(f"pairwise must have shape (N,N) or (B,N,N). Got {pw.shape}")

    B, N, _ = pw.shape
    perms = _enumerate_perms(N)

    best_scores = np.full((B,), -np.inf, dtype=np.float64)
    best_perms: List[Tuple[int, ...]] = [perms[0]] * B

    # brute force (fine for N=2/3)
    for perm in perms:
        # diag after perm: ref i matched to est perm[i]
        diag = pw[:, np.arange(N), np.array(perm)]  # (B,N)
        avg = diag.mean(axis=1)                     # (B,)
        better = avg > best_scores
        if np.any(better):
            best_scores[better] = avg[better]
            for b in np.where(better)[0]:
                best_perms[b] = perm

    # For PITResult, we return single best for single sample;
    # for batch, user can call this per-sample or use wrapper APIs.
    if B != 1:
        # Return the *average across batch* as best_score, and empty per_source by convention.
        # Prefer using higher-level APIs that return per-sample results.
        avg_score = float(best_scores.mean())
        # pick first perm for representation; not meaningful for batch
        rep_perm = best_perms[0]
        rep_diag = pw[0, np.arange(N), np.array(rep_perm)]
        return PITResult(best_perm=rep_perm, best_score=avg_score, per_source_scores=rep_diag.astype(np.float64))

    perm0 = best_perms[0]
    diag0 = pw[0, np.arange(N), np.array(perm0)]
    return PITResult(best_perm=perm0, best_score=float(best_scores[0]), per_source_scores=diag0.astype(np.float64))


def compute_pairwise(
    est: ArrayLike,
    ref: ArrayLike,
    pairwise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute pairwise score matrix using a torch pairwise_fn.

    pairwise_fn signature:
      pairwise_fn(est_1d, ref_1d) -> scalar tensor
    where est_1d/ref_1d are shape (T,).

    Returns:
      pairwise scores tensor of shape (B, N, N) on CPU.
    """
    est_t = _to_3d_torch(est, device=device)
    ref_t = _to_3d_torch(ref, device=device)
    est_t, ref_t = _crop_to_min_time(est_t, ref_t)

    if est_t.shape[:2] != ref_t.shape[:2]:
        raise ValueError(f"est/ref must match on (B,N). Got est={tuple(est_t.shape)}, ref={tuple(ref_t.shape)}")

    B, N, _ = est_t.shape
    out = torch.empty((B, N, N), dtype=torch.float32, device=est_t.device)

    for b in range(B):
        for i in range(N):      # ref index
            for j in range(N):  # est index
                out[b, i, j] = pairwise_fn(est_t[b, j], ref_t[b, i])

    return out.detach().cpu()


def pit_best_average_from_pairwise(pairwise: ArrayLike) -> Tuple[float, Tuple[int, ...], np.ndarray]:
    """
    Convenience: return best avg score, best perm, and per-source scores for single sample.
    If batch is provided, best avg score is averaged across batch (perm is representative only).
    """
    res = pit_from_pairwise_scores(pairwise)
    return res.best_score, res.best_perm, res.per_source_scores


def apply_perm_est(est: ArrayLike, perm: Sequence[int]) -> ArrayLike:
    """
    Reorder est along source dimension using perm.
    est: (N,T) or (B,N,T)
    perm: length N, mapping ref_i -> est_perm[i]
    """
    if isinstance(est, torch.Tensor):
        if est.dim() == 2:
            return est[torch.tensor(perm, device=est.device)]
        if est.dim() == 3:
            return est[:, torch.tensor(perm, device=est.device)]
        raise ValueError(f"est must be 2D or 3D, got {tuple(est.shape)}")

    if isinstance(est, np.ndarray):
        if est.ndim == 2:
            return est[np.array(perm)]
        if est.ndim == 3:
            return est[:, np.array(perm)]
        raise ValueError(f"est must be 2D or 3D, got {est.shape}")

    raise TypeError(f"est must be torch.Tensor or np.ndarray, got {type(est)}")
