# dataset_mix2_test.py
import os
import re
from typing import List, Optional, Tuple

import torch
import torchaudio


def _default_channel_indices(num_channels: int) -> Optional[List[int]]:
    if num_channels == 3:
        return [0, 2, 4]
    return None


class Mix2TestDataset(torch.utils.data.Dataset):
    """
    Evaluation dataset for Mix2 (MC-Libri2Mix / SMS-WSJ-like).

    Minimal expected structure:
      root/
        early/<split>/<base>_0.wav
        early/<split>/<base>_1.wav
        ... up to n_src-1

    Optional:
      root/
        tail/<split>/<base>_0.wav, <base>_1.wav, ...
        observation/<split>/<base>.wav   (or you can set mixture_folder="mix")
        mix/<split>/<base>.wav

    Behavior:
      - If use_mixture_file=True and <mixture_folder>/<split>/<base>.wav exists, use it as mixture.
      - Otherwise synthesize mixture = sum_k (early_k + tail_k) (tail zeros if missing).

    Returns:
      mixture: (C, T) float32
      early:   (K, C, T) float32
      tail:    (K, C, T) float32 (zeros if missing)
      utt_id:  str
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        sample_rate: int = 8000,
        n_src: int = 2,
        num_channels: int = 6,                   # 6 or 3 (3 -> [0,2,4])
        select_channels: Optional[List[int]] = None,  # override indices
        mixture_folder: str = "observation",     # "observation" or "mix"
        early_folder: str = "early",
        tail_folder: str = "tail",
        use_mixture_file: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.sample_rate = int(sample_rate)
        self.n_src = int(n_src)
        self.use_mixture_file = bool(use_mixture_file)

        if select_channels is not None:
            self.indices = list(select_channels)
        else:
            self.indices = _default_channel_indices(int(num_channels))

        self.early_dir = os.path.join(root_dir, early_folder, split)
        self.tail_dir = os.path.join(root_dir, tail_folder, split)
        self.mix_dir = os.path.join(root_dir, mixture_folder, split)

        if not os.path.isdir(self.early_dir):
            raise FileNotFoundError(f"early dir not found: {self.early_dir}")

        # Build sample list from early files: find all *_0.wav, derive base name
        prog = re.compile(r"(.+)_0\.wav$")
        bases = []
        for fn in sorted(os.listdir(self.early_dir)):
            m = prog.match(fn)
            if m:
                bases.append(m.group(1))

        if len(bases) == 0:
            raise RuntimeError(f"No '*_0.wav' found in {self.early_dir}")

        # Keep only bases that have all speakers in early
        valid = []
        for b in bases:
            ok = True
            for i in range(self.n_src):
                if not os.path.exists(os.path.join(self.early_dir, f"{b}_{i}.wav")):
                    ok = False
                    break
            if ok:
                valid.append(b)

        if len(valid) == 0:
            raise RuntimeError(f"No complete samples (0..{self.n_src-1}) found in {self.early_dir}")

        self.base_names = valid

    def __len__(self) -> int:
        return len(self.base_names)

    def _load(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (C,T)
        wav = wav.float()
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if self.indices is not None:
            wav = wav[self.indices, :]
        return wav

    def get_utt_id(self, idx: int) -> str:
        return str(self.base_names[idx])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        base = self.base_names[idx]
        utt_id = str(base)

        # ----- load early -----
        early_list = []
        for i in range(self.n_src):
            p = os.path.join(self.early_dir, f"{base}_{i}.wav")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing early: {p}")
            early_list.append(self._load(p))
        early = torch.stack(early_list, dim=0)  # (K,C,T)

        # ----- load tail (optional) -----
        tail_ok = os.path.isdir(self.tail_dir)
        if tail_ok:
            for i in range(self.n_src):
                p = os.path.join(self.tail_dir, f"{base}_{i}.wav")
                if not os.path.exists(p):
                    tail_ok = False
                    break

        if tail_ok:
            tail_list = []
            for i in range(self.n_src):
                p = os.path.join(self.tail_dir, f"{base}_{i}.wav")
                tail_list.append(self._load(p))
            tail = torch.stack(tail_list, dim=0)  # (K,C,T)
        else:
            tail = torch.zeros_like(early)

        # ----- load mixture (optional) -----
        mixture = None
        if self.use_mixture_file and os.path.isdir(self.mix_dir):
            mix_path = os.path.join(self.mix_dir, f"{base}.wav")
            if os.path.exists(mix_path):
                mixture = self._load(mix_path)

        if mixture is None:
            # synthesize mixture if file is absent or disabled
            mixture = (early + tail).sum(dim=0)  # (C,T)

        # ----- align length -----
        T = min(mixture.shape[-1], early.shape[-1], tail.shape[-1])
        mixture = mixture[..., :T]
        early = early[..., :T]
        tail = tail[..., :T]

        return mixture, early, tail, utt_id
