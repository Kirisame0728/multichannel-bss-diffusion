import os
import time
from datetime import datetime

import numpy as np
import torch
import scipy.signal as sig
import soundfile as sf

from CQT_nsgt import CQT_cpx


# -------------------------
# Helpers (lazy imports)
# -------------------------
def _require_plotly():
    import plotly  # noqa: F401
    import plotly.express as px
    import plotly.graph_objects as go  # noqa: F401
    return px


def _require_pandas():
    import pandas as pd
    return pd


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# -------------------------
# Core transforms
# -------------------------
def do_stft(noisy, clean=None, win_size=2048, hop_size=512, device="cpu", DC=True):
    """
    Applies STFT. Kept compatible with existing callers.

    Returns:
      - if clean is None: stft(noisy)
      - else: (stft(noisy), stft(clean))
    Output layout is preserved: (B, 2, F, T) after permute done below.
    """
    window = torch.hamming_window(window_length=win_size, device=noisy.device)

    # pad noisy
    noisy_pad = torch.cat(
        (noisy, torch.zeros(noisy.shape[0], win_size, device=noisy.device, dtype=noisy.dtype)), dim=1
    )
    stft_noisy = torch.stft(
        noisy_pad,
        n_fft=win_size,
        hop_length=hop_size,
        window=window,
        center=False,
        return_complex=False,
    )
    # original code permutes to (B, 2, F, T)
    stft_noisy = stft_noisy.permute(0, 3, 2, 1)

    if clean is not None:
        # keep behavior: clean padding used `device` argument previously
        clean = clean.to(device) if isinstance(device, torch.device) else clean.to(device)
        clean_pad = torch.cat(
            (clean, torch.zeros(clean.shape[0], win_size, device=clean.device, dtype=clean.dtype)), dim=1
        )
        stft_clean = torch.stft(
            clean_pad,
            n_fft=win_size,
            hop_length=hop_size,
            window=window.to(clean.device),
            center=False,
            return_complex=False,
        )
        stft_clean = stft_clean.permute(0, 3, 2, 1)

        if DC:
            return stft_noisy, stft_clean
        return stft_noisy[..., 1:], stft_clean[..., 1:]

    if DC:
        return stft_noisy
    return stft_noisy[..., 1:]


# -------------------------
# Plotting: norms / losses
# -------------------------
def plot_norms(path, normsscores, normsguides, t, name):
    px = _require_plotly()
    pd = _require_pandas()

    values = _as_numpy(t)
    df = pd.DataFrame.from_dict(
        {
            "sigma": values[0:-1],
            "score": _as_numpy(normsscores),
            "guidance": _as_numpy(normsguides),
        }
    )

    fig = px.line(df, x="sigma", y=["score", "guidance"], log_x=True, log_y=True, markers=True)
    path_to_plotly_html = os.path.join(path, f"{name}.html")
    fig.write_html(path_to_plotly_html, auto_play=False)
    return fig


def plot_loss_by_sigma(values, mean_loss_in_bins):
    px = _require_plotly()
    pd = _require_pandas()

    df = pd.DataFrame.from_dict({"sigma": values, "loss": mean_loss_in_bins})
    fig = px.line(df, x="sigma", y="loss", log_x=True, markers=True, range_y=[0, 2])
    return fig


def plot_loss_by_sigma_test(average_loss, t):
    return plot_loss_by_sigma(np.array(t), average_loss)


def plot_loss_by_sigma_train(sum_loss_in_bins, num_elems_in_bins, quantized_sigma_values):
    valid_bins = num_elems_in_bins > 0
    mean_loss_in_bins = (sum_loss_in_bins[valid_bins] / num_elems_in_bins[valid_bins])
    values = quantized_sigma_values[valid_bins]
    return plot_loss_by_sigma(values, mean_loss_in_bins)


def plot_loss_by_sigma_test_snr(average_snr, average_snr_out, t):
    px = _require_plotly()
    pd = _require_pandas()

    values = np.array(t)
    df = pd.DataFrame.from_dict({"sigma": values, "SNR": average_snr, "SNR_denoised": average_snr_out})
    fig = px.line(df, x="sigma", y=["SNR", "SNR_denoised"], log_x=True, markers=True)
    return fig


# -------------------------
# Spectral analysis plots
# -------------------------
def plot_spectral_analysis_sampling(avgspecNF, avgspecDEN, ts, fs=22050, nfft=1024):
    """
    Original version printed avgspecY but didn't define it. Fixed.
    """
    px = _require_plotly()
    pd = _require_pandas()

    T, F = avgspecNF.shape
    f = torch.arange(0, F) * fs / nfft
    f = f.unsqueeze(1).repeat(1, T).reshape(-1)

    avgspecNF_flat = avgspecNF.permute(1, 0).reshape(-1)
    avgspecDEN_flat = avgspecDEN.permute(1, 0).reshape(-1)

    ts_list = _as_numpy(ts).tolist()
    ts_list = (513 * np.array(ts_list)).tolist()

    df = pd.DataFrame.from_dict(
        {"f": _as_numpy(f), "noisy": _as_numpy(avgspecNF_flat), "denoised": _as_numpy(avgspecDEN_flat), "sigma": ts_list}
    )
    fig = px.line(df, x="f", y=["noisy", "denoised"], animation_frame="sigma", log_x=False, log_y=False, markers=False)
    return fig


def plot_spectral_analysis(avgspecY, avgspecNF, avgspecDEN, ts, fs=22050, nfft=1024):
    px = _require_plotly()
    pd = _require_pandas()

    T, F = avgspecNF.shape
    f = torch.arange(0, F) * fs / nfft
    f = f.unsqueeze(1).repeat(1, T).reshape(-1)

    avgspecY_flat = avgspecY.squeeze(0).unsqueeze(1).repeat(1, T).reshape(-1)
    avgspecNF_flat = avgspecNF.permute(1, 0).reshape(-1)
    avgspecDEN_flat = avgspecDEN.permute(1, 0).reshape(-1)

    ts_list = _as_numpy(ts).tolist()
    ts_list = (513 * np.array(ts_list)).tolist()

    df = pd.DataFrame.from_dict(
        {
            "f": _as_numpy(f),
            "y": _as_numpy(avgspecY_flat),
            "noisy": _as_numpy(avgspecNF_flat),
            "denoised": _as_numpy(avgspecDEN_flat),
            "sigma": ts_list,
        }
    )
    fig = px.line(df, x="f", y=["y", "noisy", "denoised"], animation_frame="sigma", log_x=False, log_y=False, markers=False)
    return fig


# -------------------------
# Spectrogram plotting
# -------------------------
def plot_melspectrogram(X, refr=1):
    px = _require_plotly()

    X = X.squeeze(1)
    Xn = _as_numpy(X)

    if refr is None:
        refr = np.max(np.abs(Xn)) + 1e-8

    S_db = 10 * np.log10(np.abs(Xn) / refr)
    S_db = np.transpose(S_db, (0, 2, 1))
    S_db = np.flip(S_db, axis=1)

    res = None
    for i in range(Xn.shape[0]):
        res = S_db[i] if res is None else np.concatenate((res, S_db[i]), axis=1)

    fig = px.imshow(res, zmin=-40, zmax=20)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_cpxspectrogram(X):
    px = _require_plotly()
    Xn = _as_numpy(X.squeeze(1))
    fig = px.imshow(Xn, facet_col=3, animation_frame=0)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_spectrogram(X, refr=None):
    """
    Single canonical definition (the original file had duplicates).
    Expected input: (B, 1, F, T, 2) or compatible; uses sqrt(re^2+im^2).
    """
    px = _require_plotly()

    X = X.squeeze(1)
    Xn = _as_numpy(X)
    mag = np.sqrt(Xn[:, :, :, 0] ** 2 + Xn[:, :, :, 1] ** 2)

    if refr is None:
        refr = np.max(np.abs(mag)) + 1e-8

    S_db = 10 * np.log10(np.abs(mag) / refr)
    S_db = np.transpose(S_db, (0, 2, 1))
    S_db = np.flip(S_db, axis=1)

    res = None
    for i in range(mag.shape[0]):
        res = S_db[i] if res is None else np.concatenate((res, S_db[i]), axis=1)

    fig = px.imshow(res, zmin=-40, zmax=20)
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_mag_spectrogram(X, refr=None, path=None, name="spec"):
    px = _require_plotly()
    import plotly.io  # lazy

    Xn = _as_numpy(X)
    if refr is None:
        refr = np.max(np.abs(Xn)) + 1e-8

    S_db = 10 * np.log10(np.abs(Xn) / refr)
    S_db = np.flip(S_db, axis=1)

    res = None
    for i in range(Xn.shape[0]):
        res = S_db[i] if res is None else np.concatenate((res, S_db[i]), axis=1)

    fig = px.imshow(res, zmin=-40, zmax=20)
    fig.update_layout(coloraxis_showscale=False)

    if path is not None:
        os.makedirs(path, exist_ok=True)
        plotly.io.write_image(fig, os.path.join(path, f"{name}.png"))

    return fig


# -------------------------
# Audio I/O
# -------------------------
def write_audio_file(x, sr, string: str, path="tmp/"):
    os.makedirs(path, exist_ok=True)
    out_path = os.path.join(path, f"{string}.wav")

    x = x.flatten().unsqueeze(1)
    xn = _as_numpy(x)

    peak = np.max(np.abs(xn)) if xn.size else 0.0
    if peak >= 1.0:
        xn = xn / (peak + 1e-12)

    sf.write(out_path, xn, sr)
    return out_path


# -------------------------
# CQT utilities
# -------------------------
def plot_cpxCQT_from_raw_audio(x, args, refr=None):
    fmax = args.sample_rate / 2
    fmin = fmax / (2 ** args.cqt.numocts)
    fbins = int(args.cqt.binsoct * args.cqt.numocts)
    device = x.device
    CQTransform = CQT_cpx(fmin, fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)
    X = CQTransform.fwd(x)
    return plot_cpxspectrogram(X)


def plot_CQT_from_raw_audio(x, args, refr=None):
    fmax = args.sample_rate / 2
    fmin = fmax / (2 ** args.cqt.numocts)
    fbins = int(args.cqt.binsoct * args.cqt.numocts)
    device = x.device
    CQTransform = CQT_cpx(fmin, fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    if refr is None:
        refr = 3

    X = CQTransform.fwd(x)
    return plot_spectrogram(X, refr)


# -------------------------
# STFT spectrogram from raw
# -------------------------
def get_spectrogram_from_raw_audio(x, stft, refr=1):
    X = do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X = X.permute(0, 2, 3, 1).squeeze(1)  # (B, F, T, 2)

    mag = torch.sqrt(X[:, :, :, 0] ** 2 + X[:, :, :, 1] ** 2)
    S_db = 10 * torch.log10(torch.abs(mag) / refr)

    S_db = S_db.permute(0, 2, 1)
    S_db = torch.flip(S_db, dims=[1])

    res = None
    for i in range(mag.shape[0]):
        res = S_db[i] if res is None else torch.cat((res, S_db[i]), dim=1)
    return res


def plot_spectrogram_from_raw_audio(x, stft, refr=None):
    if refr is None:
        refr = 3
    X = do_stft(x, win_size=stft.win_size, hop_size=stft.hop_size)
    X = X.permute(0, 2, 3, 1)
    return plot_spectrogram(X, refr)


def plot_spectrogram_from_cpxspec(X, refr=None):
    return plot_spectrogram(X, refr)


# -------------------------
# Misc
# -------------------------
def downsample2d(inputArray, kernelSize):
    average_kernel = np.ones((kernelSize, kernelSize))
    blurred_array = sig.convolve2d(inputArray, average_kernel, mode="same")
    return blurred_array[::kernelSize, ::kernelSize]


def print_cuda_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a
    print("memrylog", t, r, a, f)


def diffusion_spec_animation(path, x, t, stft, refr=1, name="animation_diffusion"):
    px = _require_plotly()

    Nsteps = x.shape[0]
    numsteps = 10
    tt = torch.linspace(0, Nsteps - 1, numsteps)

    i_s = []
    allX = None
    for i in tt:
        idx = int(torch.floor(i).item())
        i_s.append(idx)
        X = get_spectrogram_from_raw_audio(x[idx], stft, refr).unsqueeze(0)
        allX = X if allX is None else torch.cat((allX, X), dim=0)

    allXn = _as_numpy(allX)
    fig = px.imshow(allXn, animation_frame=0, zmin=-40, zmax=20)
    fig.update_layout(coloraxis_showscale=False)

    tvals = _as_numpy(t)[i_s]
    assert len(tvals) == len(fig.frames)

    for j, step in enumerate(fig.layout.sliders[0].steps):
        step.label = str(tvals[j])

    os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f"{name}.html"), auto_play=False)
    return fig


def diffusion_CQT_animation(path, x, t, args, refr=1, name="animation_diffusion", resample_factor=1):
    px = _require_plotly()

    Nsteps = x.shape[0]
    numsteps = 10
    tt = torch.linspace(0, Nsteps - 1, numsteps)
    i_s = []
    allX = None

    device = x.device
    fmax = args.sample_rate / 2
    fmin = fmax / (2 ** args.cqt.numocts)
    fbins = int(args.cqt.binsoct * args.cqt.numocts)
    CQTransform = CQT_cpx(fmin, fbins, args.sample_rate, args.audio_len, device=device, split_0_nyq=False)

    for i in tt:
        idx = int(torch.floor(i).item())
        i_s.append(idx)

        xx = x[idx]
        X = CQTransform.fwd(xx)
        X = torch.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2)

        S_db = 10 * torch.log10(torch.abs(X) / refr)
        S_db = S_db[:, 1:-1, 1:-1]
        S_db = S_db.permute(0, 2, 1)
        S_db = torch.flip(S_db, dims=[1])

        S_db = S_db.unsqueeze(0)
        S_db = torch.nn.functional.interpolate(
            S_db,
            size=(S_db.shape[2] // resample_factor, S_db.shape[3] // resample_factor),
            mode="bilinear",
        ).squeeze(0)

        S_dbn = _as_numpy(S_db)
        allX = S_dbn if allX is None else np.concatenate((allX, S_dbn), axis=0)

    fig = px.imshow(allX, animation_frame=0, zmin=-40, zmax=10, binary_compression_level=0)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    tvals = _as_numpy(t)[i_s]
    assert len(tvals) == len(fig.frames)

    for j, step in enumerate(fig.layout.sliders[0].steps):
        step.label = str(tvals[j])

    os.makedirs(path, exist_ok=True)
    fig.write_html(os.path.join(path, f"{name}.html"), auto_play=False)
    return fig



