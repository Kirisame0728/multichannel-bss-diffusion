# dataset_mix2_train.py
import os
import glob
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


def _default_channel_indices(num_channels: int) -> Optional[List[int]]:
    """
    Keep consistent with ArrayDPS-style convention:
      - if user requests 3ch from a 6ch recording, pick [0,2,4]
      - otherwise use all channels (None)
    """
    if num_channels == 3:
        return [0, 2, 4]
    return None


class Mix2TrainDataset(Dataset):
    """
    SMS-WSJ-like multi-channel dataset (train/dev/test).

    Expected structure:
      root/
        observation/<split>/<utt_id>.wav
        early/<split>/<utt_id>_0.wav ... <utt_id>_{K-1}.wav
        tail/<split>/<utt_id>_0.wav  ... <utt_id>_{K-1}.wav   (required if target_mode="early+late")

    Returns:
      mix: (C, T) float32
      tgt: (K, C, T) float32   (early or early+tail)
      utt_id: str
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        n_src: int = 2,
        target_mode: str = "early+late",      # "early" or "early+late"
        sample_rate: int = 8000,
        num_channels: int = 6,                # 6 or 3 (3 -> [0,2,4])
        select_channels: Optional[List[int]] = None,  # override indices
        max_len: Optional[int] = None,        # crop length in samples
        random_crop: bool = True,
        limit: Optional[int] = None,
        shuffle_files: bool = False,
        seed: int = 0,
        observation_folder: str = "observation",
        early_folder: str = "early",
        tail_folder: str = "tail",
    ):
        super().__init__()
        if target_mode not in ("early", "early+late"):
            raise ValueError("target_mode must be 'early' or 'early+late'")

        self.root_dir = root_dir
        self.split = split
        self.n_src = int(n_src)
        self.target_mode = target_mode
        self.sample_rate = int(sample_rate)
        self.max_len = max_len
        self.random_crop = bool(random_crop)

        # Channel selection
        if select_channels is not None:
            self.indices = list(select_channels)
        else:
            self.indices = _default_channel_indices(int(num_channels))

        self.obs_dir = os.path.join(root_dir, observation_folder, split)
        self.early_dir = os.path.join(root_dir, early_folder, split)
        self.tail_dir = os.path.join(root_dir, tail_folder, split)

        if not os.path.isdir(self.obs_dir):
            raise FileNotFoundError(f"Missing observation dir: {self.obs_dir}")
        if not os.path.isdir(self.early_dir):
            raise FileNotFoundError(f"Missing early dir: {self.early_dir}")
        if self.target_mode == "early+late" and (not os.path.isdir(self.tail_dir)):
            raise FileNotFoundError(f"Missing tail dir (required for early+late): {self.tail_dir}")

        obs_files = sorted(glob.glob(os.path.join(self.obs_dir, "*.wav")))
        if len(obs_files) == 0:
            raise FileNotFoundError(f"No wav found in: {self.obs_dir}")

        # optional shuffle + limit
        if shuffle_files:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(len(obs_files), generator=g).tolist()
            obs_files = [obs_files[i] for i in perm]

        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(f"limit must be positive, got {limit}")
            obs_files = obs_files[: min(limit, len(obs_files))]

        self.obs_files = obs_files

    def __len__(self) -> int:
        return len(self.obs_files)

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (C,T)
        wav = wav.float()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        if self.indices is not None:
            wav = wav[self.indices, :]
        return wav

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        obs_path = self.obs_files[idx]
        utt_id = os.path.splitext(os.path.basename(obs_path))[0]

        mix = self._load_wav(obs_path)  # (C,T)

        early_list = []
        tail_list = []
        for k in range(self.n_src):
            e_path = os.path.join(self.early_dir, f"{utt_id}_{k}.wav")
            if not os.path.exists(e_path):
                raise FileNotFoundError(f"Missing early: {e_path}")
            early_list.append(self._load_wav(e_path))

            if self.target_mode == "early+late":
                t_path = os.path.join(self.tail_dir, f"{utt_id}_{k}.wav")
                if not os.path.exists(t_path):
                    raise FileNotFoundError(f"Missing tail: {t_path}")
                tail_list.append(self._load_wav(t_path))

        early = torch.stack(early_list, dim=0)  # (K,C,T)
        if self.target_mode == "early+late":
            tail = torch.stack(tail_list, dim=0)  # (K,C,T)
            tgt = early + tail
        else:
            tgt = early

        # Align lengths
        T = min(mix.shape[-1], tgt.shape[-1])
        mix = mix[..., :T]
        tgt = tgt[..., :T]

        # Optional crop
        if self.max_len is not None and T > self.max_len:
            if self.random_crop:
                start = torch.randint(0, T - self.max_len + 1, (1,)).item()
            else:
                start = 0
            end = start + self.max_len
            mix = mix[..., start:end]
            tgt = tgt[..., start:end]

        return mix, tgt, utt_id


def pad_collate_mix2_train(batch):
    """
    Collate for Mix2TrainDataset.

    Returns:
      mix_pad: (B, C, T_max)
      tgt_pad: (B, K, C, T_max)
      ilens:   (B,)
      utt_ids: List[str]
    """
    mixes, tgts, utt_ids = zip(*batch)
    B = len(mixes)
    C = mixes[0].shape[0]
    K = tgts[0].shape[0]

    ilens = torch.tensor([m.shape[-1] for m in mixes], dtype=torch.long)
    T_max = int(ilens.max().item())

    mix_pad = torch.zeros(B, C, T_max, dtype=torch.float32)
    tgt_pad = torch.zeros(B, K, C, T_max, dtype=torch.float32)

    for i in range(B):
        T = mixes[i].shape[-1]
        mix_pad[i, :, :T] = mixes[i]
        tgt_pad[i, :, :, :T] = tgts[i]

    return mix_pad, tgt_pad, ilens, list(utt_ids)
