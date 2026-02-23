#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Separation core for Neural-FCA baseline (repo-integrated).

This file is REPLACED from the original upstream separate.py to:
- match our training checkpoint format (last.pt contains enc/dec together),
- match our STFT config (n_fft/hop/sample_rate from args),
- expose a single callable API:
    finetune(mix_ct_np, enc, dec, n_fft, hop, ...)
  returning time-domain estimates (K, T).

Upstream separate.py assumed sr=16000, n_fft=512/hop=128, and encoder.pt/decoder.pt.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import torch
from torch import nn


# -------------------------
# Numerical helpers
# -------------------------

def _hermitian(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.conj().transpose(-1, -2))


def safe_cholesky(Y: torch.Tensor, eps: float = 1e-4, max_tries: int = 6) -> torch.Tensor:
    """
    Robust batched Cholesky for Hermitian matrices.
    Adds diagonal jitter progressively until PD.
    """
    M = Y.shape[-1]
    I = torch.eye(M, dtype=Y.dtype, device=Y.device)
    Yh = _hermitian(Y)

    jitter = float(eps)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(Yh + jitter * I)
        except RuntimeError:
            jitter *= 10.0
    return torch.linalg.cholesky(Yh + jitter * I)


def logdet_hermitian_pd(Y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    log|Y| via Cholesky, stable and real-valued for (near) PD Hermitian matrices.
    Y: (..., M, M)
    Returns: (...) real
    """
    L = safe_cholesky(Y, eps=eps)
    diag = torch.diagonal(L, dim1=-2, dim2=-1).real
    return 2.0 * torch.log(diag + 1e-12).sum(dim=-1)


def inv_hermitian(Y: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Robust inverse using solve with diagonal loading and Hermitian symmetrization.
    """
    M = Y.shape[-1]
    I = torch.eye(M, dtype=Y.dtype, device=Y.device)
    Yh = _hermitian(Y) + eps * I
    # solve Yh * X = I
    return torch.linalg.solve(Yh, I)


# -------------------------
# STFT helpers (match training)
# -------------------------

def stft_mc_1utt(mix_tm: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """
    mix_tm: (T, M) real -> X: (TT, F, M) complex
    """
    T, M = mix_tm.shape
    X = []
    for ch in range(M):
        x = torch.stft(
            mix_tm[:, ch],
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
            center=True,
        )  # (F, TT)
        X.append(x)
    X = torch.stack(X, dim=-1)  # (F, TT, M)
    return X.permute(1, 0, 2).contiguous()  # (TT, F, M)


def istft_1utt(S_tf: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor, length: int) -> torch.Tensor:
    """
    S_tf: (TT, F) complex -> s_t: (T,) real
    """
    # torch.istft expects (F, TT)
    s = torch.istft(
        S_tf.permute(1, 0).contiguous(),
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=True,
        length=length,
    )
    return s


# -------------------------
# Core FCA updates
# -------------------------

def init_H(F: int, K: int, M: int, device: torch.device, dtype: torch.dtype, eps: float = 1e-4) -> torch.Tensor:
    """
    H: (F, K, M, M) complex Hermitian PD-ish
    """
    I = torch.eye(M, dtype=dtype, device=device)
    H = I[None, None].repeat(F, K, 1, 1)
    return _hermitian(H + eps * I)


def update_H_em(xx_tfmm: torch.Tensor, lm_tfk: torch.Tensor, H_fkmm: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    One EM-style update for full-rank SCMs.

    xx_tfmm: (TT, F, M, M)
    lm_tfk : (TT, F, K) real/complex (we use real)
    H_fkmm : (F, K, M, M)

    Returns:
      H_new: (F, K, M, M)
    """
    device = xx_tfmm.device
    dtype = xx_tfmm.dtype
    TT, F, M, _ = xx_tfmm.shape
    K = lm_tfk.shape[-1]

    I = torch.eye(M, dtype=dtype, device=device)
    eI = eps * I

    # Yk: (TT, F, K, M, M)
    # lm is real >=0, H is complex
    Yk = lm_tfk[..., None, None] * H_fkmm[None, :, :, :, :]  # broadcast: (TT,F,K,M,M)
    Y = Yk.sum(dim=2) + eI  # (TT, F, M, M)

    Yi = inv_hermitian(Y, eps=eps)  # (TT,F,M,M)

    # Yi * xx * Yi
    Yixx = Yi @ xx_tfmm
    YixxYi = Yixx @ Yi  # (TT,F,M,M)

    # Zk = Yk + Yk (YixxYi - Yi) Yk
    A = (YixxYi - Yi)  # (TT,F,M,M)
    Zk = Yk + (Yk @ A[:, :, None] @ Yk)  # (TT,F,K,M,M)

    # H = (1/TT) * sum_t Zk / lm
    lm_safe = lm_tfk.clamp_min(1e-10)
    H_new = (Zk / lm_safe[..., None, None]).sum(dim=0) / float(TT)  # sum over t -> (F,K,M,M)
    H_new = _hermitian(H_new + eI)
    return H_new


def nll_gaussian(xx_tfmm: torch.Tensor, lm_tfk: torch.Tensor, H_fkmm: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    NLL up to constant:
      sum_{t,f} ( log|Y| + tr(xx @ Y^{-1}) )
    """
    TT, F, M, _ = xx_tfmm.shape
    I = torch.eye(M, dtype=xx_tfmm.dtype, device=xx_tfmm.device)

    Yk = lm_tfk[..., None, None] * H_fkmm[None, :, :, :, :]
    Y = Yk.sum(dim=2) + eps * I  # (TT,F,M,M)
    Y = _hermitian(Y)

    logdet = logdet_hermitian_pd(Y, eps=eps)  # (TT,F)
    Yi = inv_hermitian(Y, eps=eps)            # (TT,F,M,M)

    tr = torch.einsum("tfmn,tfnm->tf", xx_tfmm, Yi).real
    return (logdet + tr).sum()


# -------------------------
# Public API used by infer_neural_fca.py
# -------------------------

@torch.enable_grad()
def finetune(
    mix_ct_np: np.ndarray,
    enc,
    dec,
    *,
    n_fft: int,
    hop: int,
    n_iter: int = 50,
    n_ziter: int = 1,
    n_hiter: int = 1,
    out_ch: int = 0,
    lr_z: float = 2e-1,
    eps: float = 1e-4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Args:
      mix_ct_np: (C, T) float numpy (channels already selected)
      enc/dec  : trained modules (already loaded from last.pt)
      n_fft/hop: must match training
      n_iter   : outer iterations (alternate H update / z update)
      n_ziter  : z steps per outer iter
      n_hiter  : H EM updates per outer iter
      out_ch   : which mixture channel to output (0-based in selected channels)
      lr_z     : Adam lr for z
      eps      : diagonal loading / jitter
      device   : torch device

    Returns:
      est_kt_np: (K, T) float32 numpy
    """
    assert mix_ct_np.ndim == 2, "mix_ct_np must be (C,T)"
    C, T = mix_ct_np.shape

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dtype_c = torch.complex64

    mix_tm = torch.from_numpy(mix_ct_np.T).to(device=device, dtype=torch.float32)  # (T,C)
    window = torch.hann_window(n_fft, device=device)

    # STFT: (TT,F,M)
    X = stft_mc_1utt(mix_tm, n_fft=n_fft, hop=hop, window=window).to(dtype_c)
    TT, F, M = X.shape
    assert M == C

    # normalize like training (per-utterance)
    scale = torch.sqrt(X.abs().pow(2).mean() + 1e-12)
    Xn = X / scale

    # xx: (TT,F,M,M)
    xx = Xn[..., :, None] * Xn[..., None, :].conj()

    # initial z from encoder distribution (match training usage)
    enc.eval()
    dec.eval()
    with torch.no_grad():
        q = enc(Xn[None], distribution=True)  # Normal-like
        z0 = q.loc  # (1, ..., latent...)
    z = nn.Parameter(z0.clone())
    opt = torch.optim.Adam([z], lr=float(lr_z))

    # init H
    K = int(getattr(dec, "K", None) or dec(z).shape[-1])
    H = init_H(F=F, K=K, M=M, device=device, dtype=dtype_c, eps=eps)  # (F,K,M,M)

    def decode_lm() -> torch.Tensor:
        lm = dec(z).clamp_min(1e-10)  # (1,TT,F,K) or (1,*,F,K)
        if lm.dim() == 4:
            return lm[0]              # (TT,F,K)
        raise RuntimeError(f"Unexpected lm shape from decoder: {tuple(lm.shape)}")

    lm = decode_lm()

    # alternating optimization
    for _ in range(int(n_iter)):
        # update H (EM-style)
        with torch.no_grad():
            for __ in range(int(n_hiter)):
                H = update_H_em(xx, lm, H, eps=eps)

        # update z (gradient steps)
        for __ in range(int(n_ziter)):
            # compute NLL
            loss = nll_gaussian(xx, lm, H, eps=eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            lm = decode_lm()

    # multichannel Wiener filter, take one output channel
    with torch.no_grad():
        I = torch.eye(M, dtype=dtype_c, device=device)
        Yk = lm[..., None, None] * H[None]          # (TT,F,K,M,M)
        Y = Yk.sum(dim=2) + eps * I                 # (TT,F,M,M)
        Y = _hermitian(Y)
        Yi = inv_hermitian(Y, eps=eps)              # (TT,F,M,M)

        # s_k(t,f,m) = Yk * Yi * x
        # first compute Wk = Yk @ Yi : (TT,F,K,M,M)
        Wk = Yk @ Yi[:, :, None]
        # apply to mixture vector: (TT,F,K,M)
        Sk = torch.einsum("tfkmm,tfm->tfkm", Wk, Xn)
        # take out_ch -> (K,TT,F)
        S_k_tf = Sk[..., out_ch].permute(2, 0, 1).contiguous()

        # ISTFT each source
        est = []
        for k in range(K):
            s_t = istft_1utt(S_k_tf[k] * scale, n_fft=n_fft, hop=hop, window=window, length=T)
            est.append(s_t)
        est_kt = torch.stack(est, dim=0)  # (K,T)
        est_kt = est_kt.to(torch.float32).cpu().numpy()

    return est_kt
