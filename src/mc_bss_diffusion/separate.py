# src/mc_bss_diffusion/separate.py
import os
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml
from dotmap import DotMap

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
root = _THIS_DIR
while root != root.parent and not (root / "src").is_dir():
    root = root.parent
if (root / "src").is_dir() and str(root) not in sys.path:
    sys.path.insert(0, str(root))
from src.metrics.sdr import batch_SDR_torch
from src.metrics.eval_metrics import sisdr_batch, pesq_batch, estoi_batch
from src.dataloaders.dataset_mix2_test import Mix2TestDataset


def append_per_utt_row(csv_path: str, row: dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def save_separated_samples(outs_list, save_dir: str, utt_id: str, sr: int) -> None:
    sample_dir = os.path.join(save_dir, str(utt_id))
    os.makedirs(sample_dir, exist_ok=True)

    for trial_idx, outs in enumerate(outs_list):
        trial_dir = os.path.join(sample_dir, f"trial_{trial_idx}")
        os.makedirs(trial_dir, exist_ok=True)

        n_spk, _ = outs.shape
        for spk in range(n_spk):
            save_path = os.path.join(trial_dir, f"s{spk + 1}.wav")
            torchaudio.save(save_path, outs[spk].unsqueeze(0).cpu(), sample_rate=sr)


def save_sdr_list(sdr_list, mix_snr_list, save_dir: str, max_trials: int) -> None:
    sdr_csv_path = os.path.join(save_dir, "sdr_mix_snr.csv")
    with open(sdr_csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        title = (
            ["utt_id"]
            + [f"SDR{t}" for t in range(max_trials)]
            + [f"MixSNR{t}" for t in range(max_trials)]
            + ["max_sdr", "max_mix_snr"]
        )
        writer.writerow(title)

        for (utt_id, sdr_values), (_, mix_snr_values) in zip(sdr_list, mix_snr_list):
            sdr_values = list(sdr_values) + [np.nan] * (max_trials - len(sdr_values))
            mix_snr_values = list(mix_snr_values) + [np.nan] * (max_trials - len(mix_snr_values))

            max_sdr = float(np.nanmax(sdr_values)) if len(sdr_values) else float("nan")
            max_mix_snr = float(np.nanmax(mix_snr_values)) if len(mix_snr_values) else float("nan")
            writer.writerow([utt_id] + sdr_values + mix_snr_values + [max_sdr, max_mix_snr])


def check_existing_outputs(save_dir: str, utt_id: str, num_speakers: int, max_trials: int) -> bool:
    sample_dir = os.path.join(save_dir, str(utt_id))
    if not os.path.exists(sample_dir):
        return False

    expected = [f"s{spk + 1}.wav" for spk in range(num_speakers)]

    for trial_idx in range(max_trials):
        trial_dir = os.path.join(sample_dir, f"trial_{trial_idx}")
        if not os.path.exists(trial_dir):
            return False

        actual = set(os.listdir(trial_dir))
        if not all(fn in actual for fn in expected):
            return False

    return True


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"count": 0.0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    if a.size == 1:
        return {"count": float(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float("nan")}
    return {"count": float(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std(ddof=1))}


def concatenate_params(_args_dict) -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def modify_and_create_save_dir(config: dict) -> None:
    num_speakers = config["num_speakers"]
    reverb = "reverb" if config["reverb"] else "anechoic"
    architecture = config["architecture"]
    blind_or_oracle = "blind" if config["blind"] else "oracle"

    folder_name = concatenate_params(config)
    new_save_dir = os.path.join(
        config["save_dir"],
        f"{num_speakers}speaker_{reverb}",
        architecture,
        blind_or_oracle,
        folder_name,
    )
    config["save_dir"] = new_save_dir
    os.makedirs(new_save_dir, exist_ok=True)
    print(f"[save_dir] {new_save_dir}")


def run_inference(dataset, sampler, device, save_dir: str, cfg, n_samples: int, start_sample: int = 0) -> None:
    sdr_list = []
    mixture_snr_list = []

    per_utt_path = os.path.join(save_dir, "per_utt.csv")
    summary_path = os.path.join(save_dir, "summary.csv")

    agg = {"best_sdr": [], "best_sisdr": [], "best_pesq": [], "best_estoi": []}

    for i, (mixture, early, tail, utt_id) in enumerate(dataset):
        if i < start_sample:
            continue
        if (i - start_sample) >= n_samples:
            break

        sources = early + tail
        utt_id = str(utt_id)

        if cfg.skip_if_exists and check_existing_outputs(save_dir, utt_id, cfg.num_speakers, cfg.max_trials3):
            continue

        if cfg.architecture == "unet_1d_att":
            sig_len = mixture.shape[-1]
            target_len = int(2 ** np.ceil(np.log2(sig_len)))
            mixture = torch.nn.functional.pad(mixture, (0, target_len - sig_len))
            sources = torch.nn.functional.pad(sources, (0, target_len - sig_len))

        mix_in = mixture.unsqueeze(0).to(device)
        mixture_dev = mixture.to(device)
        sources_dev = sources.to(device)

        current_sdrs = []
        current_mixsnrs = []
        current_outs = []

        best_sdr = -1e9
        best_trial = -1
        max_sdr_seen = -1e9
        n_trials = 0

        while True:
            outs = sampler.separate(mix_in, cfg.num_speakers, device)

            if cfg.architecture == "unet_1d_att":
                outs = outs[..., :sig_len]
                mixture_eval = mixture_dev[..., :sig_len]
                sources_eval = sources_dev[..., :sig_len]
            else:
                mixture_eval = mixture_dev
                sources_eval = sources_dev

            separated_sources = outs.reshape(1, cfg.num_speakers, -1)
            _, separated_sources_mc = sampler.spatialization(separated_sources, mixture_eval.unsqueeze(0))

            sdr = batch_SDR_torch(separated_sources_mc[:, :, 0, :], sources_eval[:, 0, :].unsqueeze(0))
            sdr_val = float(sdr.item())

            mix_snr = sampler.spatialization.calc_snr(
                mixture_eval[0].detach(), separated_sources_mc[0, :, 0, :].sum(0).detach()
            )
            mix_snr_val = float(mix_snr)

            print(f"utt_id={utt_id} idx={i} trial={n_trials} SDR={sdr_val:.3f} MixSNR={mix_snr_val:.3f}")

            current_outs.append(outs.squeeze(0).detach().cpu())
            current_sdrs.append(sdr_val)
            current_mixsnrs.append(mix_snr_val)

            if sdr_val > best_sdr:
                best_sdr = sdr_val
                best_trial = n_trials

            max_sdr_seen = max(max_sdr_seen, sdr_val)
            n_trials += 1

            if max_sdr_seen >= cfg.snr_stop:
                break
            elif max_sdr_seen >= cfg.snr_stop2 and n_trials >= cfg.max_trials:
                break
            elif max_sdr_seen >= cfg.snr_stop3 and n_trials >= cfg.max_trials2:
                break
            elif n_trials >= cfg.max_trials3:
                break

        save_separated_samples(current_outs, save_dir, utt_id, sr=cfg.sample_rate)

        best_outs = current_outs[best_trial].to(device)
        _, best_mc = sampler.spatialization(best_outs.unsqueeze(0), mixture_eval.unsqueeze(0))

        est_ch0 = best_mc[0, :, 0, :].detach().cpu()
        ref_ch0 = sources_eval[:, 0, :].detach().cpu()

        best_sisdr = float(sisdr_batch(est_ch0, ref_ch0))
        best_pesq = float(pesq_batch(est_ch0.numpy(), ref_ch0.numpy(), sr=cfg.sample_rate))
        best_estoi = float(estoi_batch(est_ch0.numpy(), ref_ch0.numpy(), sr=cfg.sample_rate))

        append_per_utt_row(
            per_utt_path,
            {
                "utt_id": utt_id,
                "num_trials": int(n_trials),
                "best_trial": int(best_trial),
                "best_sdr": float(best_sdr),
                "best_sisdr": float(best_sisdr),
                "best_pesq": float(best_pesq),
                "best_estoi": float(best_estoi),
            },
        )

        agg["best_sdr"].append(best_sdr)
        agg["best_sisdr"].append(best_sisdr)
        agg["best_pesq"].append(best_pesq)
        agg["best_estoi"].append(best_estoi)

        sdr_list.append((utt_id, current_sdrs))
        mixture_snr_list.append((utt_id, current_mixsnrs))

    save_sdr_list(sdr_list, mixture_snr_list, save_dir, cfg.max_trials3)

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "count", "mean", "median", "std"])
        for m in ["best_sdr", "best_sisdr", "best_pesq", "best_estoi"]:
            s = summarize(agg[m])
            if int(s["count"]) == 0:
                continue
            w.writerow([m, int(s["count"]), s["mean"], s["median"], s["std"]])


def main():
    parser = argparse.ArgumentParser(description="mc_bss_diffusion separator")

    parser.add_argument("--config_path", type=str, default="configs/mc_bss_diffusion/conf_libritts_unet1d_attention_8k.yaml")
    parser.add_argument("--architecture", type=str, default="unet_1d")
    parser.add_argument("--checkpoint", type=str, default="weights-459999.pt")

    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_speakers", type=int, default=2)
    parser.add_argument("--n_channels", type=int, default=6)
    parser.add_argument("--mixture_folder", type=str, default="observation")
    parser.add_argument("--use_mixture_file", type=lambda x: bool(int(x)), default=True)

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=2)
    parser.add_argument("--start_sample", type=int, default=0)
    parser.add_argument("--blind", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--reverb", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--skip_if_exists", type=lambda x: bool(int(x)), default=True)

    parser.add_argument("--num_steps", type=int, default=350)
    parser.add_argument("--max_trials", type=int, default=2)
    parser.add_argument("--snr_stop", type=float, default=18.0)
    parser.add_argument("--max_trials2", type=int, default=3)
    parser.add_argument("--snr_stop2", type=float, default=13.0)
    parser.add_argument("--max_trials3", type=int, default=4)
    parser.add_argument("--snr_stop3", type=float, default=10.0)

    parser.add_argument("--sigma_min", type=float, default=1e-4)
    parser.add_argument("--sigma_max", type=float, default=8.0)
    parser.add_argument("--rho", type=int, default=10)
    parser.add_argument("--schurn", type=float, default=30.0)
    parser.add_argument("--xi", type=float, default=1.3)

    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--lambda_reg", type=float, default=1e-3)
    parser.add_argument("--n_frames_past", type=int, default=20)
    parser.add_argument("--n_frames_future", type=int, default=0)
    parser.add_argument("--fcp_epsilon", type=float, default=1e-2)

    parser.add_argument("--ref_loss_weight", type=float, default=0.3)
    parser.add_argument("--ref_loss_snr_threshold", type=float, default=20.0)
    parser.add_argument("--ref_loss_max_step", type=int, default=100)

    parser.add_argument("--use_warm_initialization", type=lambda x: bool(int(x)), default=True)
    parser.add_argument("--warm_initialization_rescale", type=lambda x: bool(int(x)), default=False)
    parser.add_argument("--warm_initialization_sigma", type=float, default=0.057)
    parser.add_argument("--initialized_filter_step", type=int, default=200)

    cfg = parser.parse_args()

    cfg.sample_rate = 8000

    args_dict = vars(cfg)
    modify_and_create_save_dir(args_dict)
    cfg.save_dir = args_dict["save_dir"]

    with open(os.path.join(cfg.save_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    args = yaml.safe_load(Path(cfg.config_path).read_text())
    args = DotMap(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dirname = os.getcwd()
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    os.makedirs(args.model_dir, exist_ok=True)

    args.architecture = cfg.architecture
    args.inference.checkpoint = cfg.checkpoint
    args.sample_rate = cfg.sample_rate

    args.inference.T = cfg.num_steps
    args.diffusion_parameters.sigma_min = cfg.sigma_min
    args.diffusion_parameters.sigma_max = cfg.sigma_max
    args.diffusion_parameters.ro = cfg.rho
    args.inference.xi = cfg.xi
    args.diffusion_parameters.sigma_data = cfg.warm_initialization_sigma

    from utils.setup import load_ema_weights

    if args.architecture == "unet_1d":
        from models.unet_1d import Unet_1d
        model = Unet_1d(args, device).to(device)
    elif args.architecture == "unet_1d_att":
        from models.unet_1d_attn import UNet1dAttn
        model = UNet1dAttn(args.unet_wav, device).to(device)
    else:
        raise NotImplementedError(args.architecture)

    model = load_ema_weights(model, os.path.join(args.model_dir, args.inference.checkpoint))
    print(f"[ckpt] loaded: {os.path.join(args.model_dir, args.inference.checkpoint)}")

    dataset = Mix2TestDataset(
        root_dir=cfg.root_dir,
        split=cfg.split,
        sample_rate=cfg.sample_rate,
        n_src=cfg.num_speakers,
        num_channels=cfg.n_channels,
        mixture_folder=cfg.mixture_folder,
        use_mixture_file=cfg.use_mixture_file,
    )

    from sampler_spatial_v1_reverb_iva_8kHz import Sampler
    from sde import VE_Sde_Elucidating

    diff_parameters = VE_Sde_Elucidating(args.diffusion_parameters, args.diffusion_parameters.sigma_data)
    sampler = Sampler(
        model,
        diff_parameters,
        args,
        args.inference.xi,
        order=2,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.n_fft,
        lambda_reg=cfg.lambda_reg,
        n_frames_past=cfg.n_frames_past,
        n_frames_future=cfg.n_frames_future,
        fcp_epsilon=cfg.fcp_epsilon,
        n_spks=cfg.num_speakers,
        use_warm_initialization=cfg.use_warm_initialization,
        warm_initialization_rescale=cfg.warm_initialization_rescale,
        warm_initialization_sigma=cfg.warm_initialization_sigma,
        initialized_filter_step=cfg.initialized_filter_step,
        ref_loss_weight=cfg.ref_loss_weight,
        ref_loss_snr_threshold=cfg.ref_loss_snr_threshold,
        ref_loss_max_step=cfg.ref_loss_max_step,
    )

    run_inference(
        dataset,
        sampler,
        device,
        cfg.save_dir,
        cfg,
        n_samples=cfg.n_samples,
        start_sample=cfg.start_sample,
    )


if __name__ == "__main__":
    main()