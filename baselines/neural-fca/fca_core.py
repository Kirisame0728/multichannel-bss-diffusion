import torch

def safe_cholesky(Y: torch.Tensor, eps: float = 1e-4, max_tries: int = 6) -> torch.Tensor:
    """
    Robust Cholesky for batched Hermitian matrices.
    If not PD, increase diagonal loading (jitter) progressively.
    """
    M = Y.shape[-1]
    I = torch.eye(M, dtype=Y.dtype, device=Y.device)
    jitter = eps

    # ensure Hermitian to reduce numerical issues
    Yh = 0.5 * (Y + Y.conj().transpose(-1, -2))

    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(Yh + jitter * I)
        except RuntimeError:
            jitter *= 10.0

    # If still failing, do one last attempt with a large jitter.
    return torch.linalg.cholesky(Yh + jitter * I)

def regularized_inverse(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M = A.shape[-1]
    I = torch.eye(M, dtype=A.dtype, device=A.device)
    return torch.linalg.inv(A + eps * I)


def init_H(F: int, K: int, M: int, device, dtype=torch.complex64) -> torch.Tensor:
    I = torch.eye(M, dtype=dtype, device=device)
    return I[None, None, :, :].repeat(F, K, 1, 1).contiguous()


def build_Y(lm: torch.Tensor, H: torch.Tensor, eps: float = 1e-4):
    """
    lm: (T,F,K) >=0
    H:  (F,K,M,M) complex
    """
    Yk = lm[:, :, :, None, None] * H[None, :, :, :, :]   # (T,F,K,M,M)
    Y = Yk.sum(dim=2)                                    # (T,F,M,M)
    M = Y.shape[-1]
    I = torch.eye(M, dtype=Y.dtype, device=Y.device)
    Y = Y + eps * I[None, None, :, :]
    Y = 0.5 * (Y + Y.conj().transpose(-1, -2))
    return Y, Yk


def nll_gaussian(xx: torch.Tensor, lm: torch.Tensor, H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    xx: (T,F,M,M) = x x^H
    """
    Y, _ = build_Y(lm, H, eps=eps)
    Yi = regularized_inverse(Y, eps=eps)
    # logdet(Y) via Cholesky: stable and real for Hermitian PD matrices
    # L = torch.linalg.cholesky(Y)  # (T,F,M,M)
    L = safe_cholesky(Y, eps=eps, max_tries=6)
    diag = torch.diagonal(L, dim1=-2, dim2=-1).real
    logdet = 2.0 * torch.log(diag + 1e-12).sum(dim=-1)  # (T,F)

    tr = torch.einsum("tfmn,tfmn->tf", xx, Yi).real
    return (logdet + tr).sum()


@torch.no_grad()
def update_H_em(xx: torch.Tensor, lm: torch.Tensor, H: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    One FCA-style EM update for full-rank SCM.
    """
    T, F, M, _ = xx.shape
    K = lm.shape[-1]

    Y, _ = build_Y(lm, H, eps=eps)
    Yi = regularized_inverse(Y, eps=eps)                 # (T,F,M,M)

    tmp = torch.matmul(torch.matmul(Yi, xx), Yi) - Yi     # (T,F,M,M)

    # Z[f,k] = mean_t [ lm[t,f,k] * (tmp[t,f] @ H[f,k]) ]
    tmp_lm = tmp[:, :, None, :, :] * lm[:, :, :, None, None]     # (T,F,K,M,M)
    tmpH = torch.matmul(tmp_lm, H[None, :, :, :, :])             # (T,F,K,M,M)
    Z = tmpH.mean(dim=0)                                         # (F,K,M,M)

    I = torch.eye(M, dtype=H.dtype, device=H.device)
    Z = Z + eps * I[None, None, :, :]

    Zi = torch.linalg.inv(Z)
    H_new = torch.matmul(torch.matmul(Zi, H), Zi.conj().transpose(-1, -2))

    tr = torch.real(torch.diagonal(H_new, dim1=-2, dim2=-1).sum(dim=-1))  # (F,K)
    H_new = H_new / (tr[:, :, None, None] + 1e-12)
    return H_new