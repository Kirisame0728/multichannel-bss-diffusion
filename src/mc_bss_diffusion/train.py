"""

Main script for training
"""
import os
import hydra

import torch
torch.cuda.empty_cache()

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np

import sys
from pathlib import Path

# Make repo root importable so "import src...." works when running inside src/mc_bss_diffusion/
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../<repo>/
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run(args):
    """Loads all the modules and starts the training
        
    Args:
      args:
        Hydra dictionary

    """
        
    #some preparation of the hydra args
    # args = OmegaConf.structured(OmegaConf.to_yaml(args))
    args = OmegaConf.to_container(args, resolve=True)
    args = OmegaConf.create(args)

    #choose gpu as the device if possible
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model_dir = str(args.model_dir)
    os.makedirs(args.model_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
        
    def worker_init_fn(worker_id):                                                          
        st=np.random.get_state()[2]
        np.random.seed( st+ worker_id)

    # print("Training on: ",args.dset.name)

    #prepare the dataset loader
    from src.dataloaders import dataset_libritts as dataset
    args.dataset.root = hydra.utils.to_absolute_path(args.dataset.root)
    dataset_train=dataset.LIBRITTS_TrainSet(

        root=args.dataset.root,
        # urls=['dev-other'],
        download=False,
        audio_len=args.dataset.audio_len,
        min_audio_len=args.dataset.min_audio_len,
        target_sampling_rate=args.dataset.target_sampling_rate, 
        std_norm=args.dataset.std_norm, 
        std=args.dataset.std
                               
    )
    train_loader=DataLoader(dataset.LIBRITTS_IterableDataset(dataset_train),num_workers=args.num_workers, batch_size=args.batch_size,  worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)
        
    #prepare the model architecture

    if args.architecture=="unet_1d":
        from models.unet_1d import Unet_1d
        model=Unet_1d(args, device).to(device)
    elif args.architecture=="unet_1d_att":
        from models.unet_1d_attn import UNet1dAttn
        model = UNet1dAttn(args.unet_wav, device).to(device)
    else:
        raise NotImplementedError

    #prepare the optimizer

    from learner import Learner
    
    learner = Learner(
        args.model_dir, model, train_set,  args, log=args.log
    )

    #start the training
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf_libritts_ncsnpp")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()
