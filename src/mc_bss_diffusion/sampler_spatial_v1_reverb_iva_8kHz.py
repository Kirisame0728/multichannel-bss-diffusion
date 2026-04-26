from tqdm import tqdm
import torch
import torchaudio
import scipy.signal
import numpy as np
import utils.logging as utils_logging

# this is a third version that joinly optimizes all speakers steering vector
import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.FCP import FCP_torch_2, compute_inverse_power
# from src.FCP import FCP_V1, FCP_V2
from FCP import FCP_V1
from IVA import IVA


class Sampler():

    def __init__(self,
                 model,
                 diff_params,
                 args,
                 xi=0,
                 order=2,
                 n_fft=512,
                 hop_length=128,
                 win_length=512,
                 lambda_reg=0,
                 n_frames_past=0,
                 n_frames_future=0,
                 fcp_epsilon=1e-2,
                 n_spks=2,
                 n_ch=3,

                 # use_warm_initialization=False, initialized_filter_step=0 -> not using IVA initialization
                 # use_warm_initialization=True, initialized_filter_step=0 -> only use source warm initialization
                 # use_warm_initialization=False, initialized_filter_step=200 -> only use IVA implied steering filter, need to tune number of steps
                 # use_warm_initialization=False, initialized_filter_step=total_steps -> only use IVA implied steering filter for all steps
                 # use_warm_initialization=True, initialized_filter_step=total_steps -> fully use initialization
                 use_warm_initialization=True,
                 warm_initialization_rescale=False,
                 warm_initialization_sigma=0.057,
                 initialized_filter_step=100,  # set to zero if not using IVA implied FCP filter

                 ref_loss_weight=0.5,
                 ref_loss_snr_threshold=0,  # 3dB of for optimiation
                 ref_loss_max_step=100,  # do not have ref channel loss when steps >= ref_loss_max_step
                 ):

        # things needed for IVA initialization
        self.use_warm_initialization = use_warm_initialization
        self.warm_initialization_rescale = warm_initialization_rescale
        self.warm_initialization_sigma = warm_initialization_sigma
        self.initialized_filter_step = initialized_filter_step

        self.n_ch = n_ch
        self.n_spks = n_spks
        self.single_channel_mode = (n_ch == 1)

        self.ref_loss_weight = ref_loss_weight
        self.ref_loss_snr_threshold = ref_loss_snr_threshold
        self.ref_loss_max_step = ref_loss_max_step

        self.fcp_epsilon = fcp_epsilon

        self.iva_in = min(self.n_ch, n_spks + 1)

        if not self.single_channel_mode:
            self.IVA = IVA(
                n_fft=2048,
                hop_length=256,
                win_length=2048,
                n_iter=100,
                n_src_iva=min(self.n_ch, n_spks + 1),
                n_outs=n_spks,
                proj_back_mic=0,
                model='gauss',  # gauss or laplace or nmf
                use_tiss=True,
                blank_frequency_start=1025
            )
        else:
            self.IVA = None
            self.use_warm_initialization = False

        self.model = model
        self.diff_params = diff_params
        self.xi = xi  # hyperparameter for the reconstruction guidance
        self.order = order
        self.args = args
        self.nb_steps = self.args.inference.T

        self.treshold_on_grads = args.inference.max_thresh_grads

        # add steering vector estimation and mixture resynthesis
        self.spatialization = FCP_V1(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            lambda_reg=lambda_reg,
            n_frames_past=n_frames_past,
            n_frames_future=n_frames_future,
            fcp_epsilon=self.fcp_epsilon
        )

    def get_score_rec_guidance(self, x, y, t_i, step, G_conj=None):
        # x: bs*n_spk, T
        # y: bs, n_ch, T
        x.requires_grad_()
        x_hat = self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        bs = y.shape[0]
        n_spk = int(x.shape[0] / bs)

        anchor_sources = x_hat.reshape(bs, n_spk, -1)
        anchor_mixture = anchor_sources.sum(1)  # (bs, T)

        if self.single_channel_mode or y.shape[1] == 1:
            # Single-channel fallback: only enforce mixture consistency.
            loss = (anchor_mixture - y[:, 0, :]).square().sum(-1).mean()
        else:
            _, all_channels_hat = self.spatialization(anchor_sources, y, G_conj)
            den_rec = all_channels_hat.sum(1)

            loss_all_channels = (den_rec - y).square().sum(-1).mean(1)  # (bs,)
            loss_ref_channel = (anchor_mixture - y[:, 0, :]).square().sum(-1)  # (bs,)

            snr = self.spatialization.calc_snr(y[:, 0, :], anchor_mixture).mean()
            if snr < self.ref_loss_snr_threshold and step < self.ref_loss_max_step:
                loss = (loss_all_channels + self.ref_loss_weight * loss_ref_channel).mean()
            else:
                loss = loss_all_channels.mean()

        rec_grads = torch.autograd.grad(outputs=loss, inputs=x)[0]

        normguide = torch.linalg.norm(rec_grads) / x.shape[-1] ** 0.5

        # normalize scaling
        s = self.xi / (normguide * t_i + 1e-6)

        # optionally apply a threshold to the gradients
        if self.treshold_on_grads > 0:
            rec_grads = torch.clip(rec_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)

        score = (x_hat.detach() - x) / t_i ** 2

        # apply scaled guidance to the score
        score = score - s * rec_grads

        return score

    def get_score(self, x, y, t_i, step, G_conj=None):
        if y is None:
            # unconditional sampling
            with torch.no_grad():
                x_hat = self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))
                score = (x_hat - x) / t_i ** 2
            return score
        else:
            score = self.get_score_rec_guidance(x, y, t_i, step, G_conj)
            return score

    def predict_unconditional(self, shape, device):
        self.y = None
        return self.predict(shape, device)

    def iva_initialize(self, y, sigma):
        # y: bs, n_ch, T
        bs, n_ch, T = y.shape

        if self.single_channel_mode or n_ch == 1:
            # Single-channel fallback: initialize from the reference mixture channel.
            mix_ref = y[:, 0, :].unsqueeze(1).repeat(1, self.n_spks, 1) / self.n_spks
            x = mix_ref + torch.randn_like(mix_ref) * sigma
            return x, None

        iva_outs = self.IVA(y)  # bs, n_spk, T
        if not self.use_warm_initialization:
            return torch.randn_like(iva_outs) * sigma, iva_outs
        if self.warm_initialization_rescale:
            x = self.warm_initialization_sigma * iva_outs / (iva_outs.std(-1, keepdim=True) + 1e-8)
            x = x + torch.randn_like(x) * sigma
        else:
            x = iva_outs
            x = x + torch.randn_like(x) * sigma

        return x, iva_outs

    def separate(self, y, n_spk, device):
        self.y = y
        # get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        # sample from gaussian distribution with sigma_max variance
        bs, n_ch, T = y.shape
        shape = (bs * n_spk, T)

        # use iva to warm initialize the diffusion sampling
        x, iva_out = self.iva_initialize(y, t[0])
        x = x.reshape(bs * n_spk, -1)

        # calculate the initialized FCP filter only for multichannel case
        if self.single_channel_mode or n_ch == 1:
            G_conj = None
        else:
            G_conj, _ = self.spatialization(iva_out, y)

        # parameter for langevin stochasticity
        gamma = self.diff_params.get_gamma(t).to(device)

        for i in tqdm(range(0, self.nb_steps, 1)):
            if gamma[i] == 0:
                t_hat = t[i]
                x_hat = x
            else:
                t_hat = t[i] + gamma[i] * t[i]
                epsilon = torch.randn(shape).to(device) * self.diff_params.Snoise
                x_hat = x + ((t_hat ** 2 - t[i] ** 2) ** (1 / 2)) * epsilon

            if i >= self.initialized_filter_step:
                score = self.get_score(x_hat, self.y, t_hat, i)
            else:
                score = self.get_score(x_hat, self.y, t_hat, i, G_conj)

            d = -t_hat * score
            h = t[i + 1] - t_hat

            if t[i + 1] != 0 and self.order == 2:
                t_prime = t[i + 1]
                x_prime = x_hat + h * d
                if i >= self.initialized_filter_step:
                    score = self.get_score(x_prime, self.y, t_prime, i)
                else:
                    score = self.get_score(x_prime, self.y, t_prime, i, G_conj)

                d_prime = -t_prime * score
                x = (x_hat + h * ((1 / 2) * d + (1 / 2) * d_prime))

            elif t[i + 1] == 0 or self.order == 1:
                x = x_hat + h * d

        return x.detach().view(bs, n_spk, -1)

    def predict(self, shape, device):
        # get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)
        # sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape, t[0]).to(device)

        # parameter for langevin stochasticity
        gamma = self.diff_params.get_gamma(t).to(device)

        for i in tqdm(range(0, self.nb_steps, 1)):
            if gamma[i] == 0:
                t_hat = t[i]
                x_hat = x
            else:
                t_hat = t[i] + gamma[i] * t[i]
                epsilon = torch.randn(shape).to(device) * self.diff_params.Snoise
                x_hat = x + ((t_hat ** 2 - t[i] ** 2) ** (1 / 2)) * epsilon

            score = self.get_score(x_hat, self.y, t_hat, i)
            d = -t_hat * score
            h = t[i + 1] - t_hat

            if t[i + 1] != 0 and self.order == 2:
                t_prime = t[i + 1]
                x_prime = x_hat + h * d
                score = self.get_score(x_prime, self.y, t_prime, i)
                d_prime = -t_prime * score
                x = (x_hat + h * ((1 / 2) * d + (1 / 2) * d_prime))

            elif t[i + 1] == 0 or self.order == 1:
                x = x_hat + h * d

        return x.detach()
