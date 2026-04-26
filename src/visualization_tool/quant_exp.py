import os
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import stft


# =========================
# 1. Config
# =========================
case_name = "case_A_medium-difficulty"

audio_paths = {
    "Observed mixture": "D:/datasets/libri_test_2sp_6ch_8k/observation/test/1580-141083-0044_8555-284449-0016.wav",
    "Proposed": "D:/datasets/libri_separation_outputs/2speaker_reverb/unet_1d_att/blind/20260208_200544/1580-141083-0044_8555-284449-0016/trial_0/s2.wav",
    "AuxIVA (Laplacian)": "D:/datasets/libri_separation_outputs/iva-laplacian/1580-141083-0044_8555-284449-0016/s1.wav",
        "AuxIVA (Gaussian)": "D:/datasets/libri_separation_outputs/iva-gaussian/1580-141083-0044_8555-284449-0016/s1.wav",
    "TF-GridNet": "D:/datasets/libri_separation_outputs/tfgridnet/1580-141083-0044_8555-284449-0016/s2.wav",
    "Neural-FCA": "D:/datasets/libri_separation_outputs/neural_fca/1580-141083-0044_8555-284449-0016/s2.wav",
    "FastMNMF2": "D:/datasets/libri_separation_outputs/fastmnmf2-dp_retrained_vae_neweval/1580-141083-0044_8555-284449-0016/s1.wav",
}

# reference = early + late
reference_early_path = "D:/datasets/libri_test_2sp_6ch_8k/early/test/1580-141083-0044_8555-284449-0016_1.wav"
reference_late_path = "D:/datasets/libri_test_2sp_6ch_8k/tail/test/1580-141083-0044_8555-284449-0016_1.wav"

output_dir = Path("qualitative_figures")
output_dir.mkdir(exist_ok=True)

# 统一显示的时间范围
start_sec = 0.0
duration_sec = 3.0

# STFT 参数
n_fft = 1024
hop_length = 256
win_length = 1024

# 频谱图显示范围
max_freq_hz = 4000

# dB dynamic range
db_min = -80
db_max = 0

# 如果音频是多通道，选哪个通道；None 表示取均值
channel_index = 0


# =========================
# 2. Helpers
# =========================
def load_audio(path, channel_index=0):
    wav, sr = sf.read(path, always_2d=True)  # shape: (T, C)
    if channel_index is None:
        wav = wav.mean(axis=1)
    else:
        channel_index = min(channel_index, wav.shape[1] - 1)
        wav = wav[:, channel_index]
    return wav.astype(np.float32), sr


def crop_audio(wav, sr, start_sec, duration_sec):
    start = int(start_sec * sr)
    end = int((start_sec + duration_sec) * sr)
    end = min(end, len(wav))
    return wav[start:end]


def compute_stft_db(wav, sr, n_fft, hop_length, win_length):
    f, t, Zxx = stft(
        wav,
        fs=sr,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx)
    mag_db = 20 * np.log10(np.maximum(mag, 1e-8))
    mag_db = np.clip(mag_db, db_min, db_max)
    return f, t, mag_db


# =========================
# 3. Load all audios
# =========================
signals = {}
sample_rates = []

# load normal signals
for label, path in audio_paths.items():
    wav, sr = load_audio(path, channel_index=channel_index)
    wav = crop_audio(wav, sr, start_sec, duration_sec)
    signals[label] = wav
    sample_rates.append(sr)

# load reference = early + late
ref_early, sr_early = load_audio(reference_early_path, channel_index=channel_index)
ref_late, sr_late = load_audio(reference_late_path, channel_index=channel_index)

if sr_early != sr_late:
    raise ValueError(f"Reference early/late sample rates do not match: {sr_early} vs {sr_late}")

ref_early = crop_audio(ref_early, sr_early, start_sec, duration_sec)
ref_late = crop_audio(ref_late, sr_late, start_sec, duration_sec)

ref_len = min(len(ref_early), len(ref_late))
reference = ref_early[:ref_len] + ref_late[:ref_len]

signals["Reference source"] = reference
sample_rates.append(sr_early)

if len(set(sample_rates)) != 1:
    raise ValueError(f"All sample rates must be the same, but got: {set(sample_rates)}")

sr = sample_rates[0]

# 按你想显示的顺序重新排一下
ordered_labels = [
    "Observed mixture",
    "Reference source",
    "Proposed",
    "AuxIVA (Laplacian)",
    "AuxIVA (Gaussian)",
    "TF-GridNet",
    "Neural-FCA",
    "FastMNMF2",
]
signals = {k: signals[k] for k in ordered_labels}

# 统一长度，防止显示时间范围不一致
min_len = min(len(x) for x in signals.values())
for k in signals:
    signals[k] = signals[k][:min_len]

time_axis = np.arange(min_len) / sr


# =========================
# 4. Plot waveform comparison
# =========================
n_rows = len(signals)

fig, axes = plt.subplots(n_rows, 1, figsize=(9, 1.6 * n_rows), dpi=300, sharex=True)

if n_rows == 1:
    axes = [axes]

for ax, (label, wav) in zip(axes, signals.items()):
    ax.plot(time_axis, wav, color="#2F5D8A", linewidth=1.0)
    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#D9D9D9", alpha=0.7)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Waveform comparison: {case_name}", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])

wave_png = output_dir / f"waveform_{case_name}.png"
wave_pdf = output_dir / f"waveform_{case_name}.pdf"
fig.savefig(wave_png, bbox_inches="tight")
fig.savefig(wave_pdf, bbox_inches="tight")
plt.show()


# =========================
# 5. Plot spectrogram comparison
# =========================
fig, axes = plt.subplots(n_rows, 1, figsize=(9, 1.8 * n_rows), dpi=300, sharex=True)

if n_rows == 1:
    axes = [axes]

im = None

for ax, (label, wav) in zip(axes, signals.items()):
    f, t, S_db = compute_stft_db(wav, sr, n_fft, hop_length, win_length)

    freq_mask = f <= max_freq_hz
    f_show = f[freq_mask]
    S_show = S_db[freq_mask, :]

    im = ax.imshow(
        S_show,
        origin="lower",
        aspect="auto",
        extent=[t[0], t[-1] if len(t) > 0 else duration_sec, f_show[0], f_show[-1]],
        vmin=db_min,
        vmax=db_max,
    )

    ax.set_ylabel(label, rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[-1].set_xlabel("Time (s)")
fig.suptitle(f"Spectrogram comparison: {case_name}", y=0.995)

# 先给右侧 colorbar 预留空间
fig.tight_layout(rect=[0, 0, 0.93, 0.98])

# 单独放 colorbar，避免与子图重叠
cbar_ax = fig.add_axes([0.935, 0.12, 0.015, 0.76])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Magnitude (dB)")

spec_png = output_dir / f"spectrogram_{case_name}.png"
spec_pdf = output_dir / f"spectrogram_{case_name}.pdf"
fig.savefig(spec_png, bbox_inches="tight")
fig.savefig(spec_pdf, bbox_inches="tight")
plt.show()

print(f"Saved: {wave_png}")
print(f"Saved: {wave_pdf}")
print(f"Saved: {spec_png}")
print(f"Saved: {spec_pdf}")