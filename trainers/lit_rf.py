import copy
from collections import OrderedDict
from utils.common import frozen_module

import torch
from torch import nn
import torchmetrics
import wandb
from utils.common import instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt, calculate_ssim_pt, calculate_kld, calculate_alkd
from utils.interpolation import linear_interpolate, natural_cubic_spline_interpolate
from utils.t_dist import ExponentialPDF, sample_t, sample_t_uniform
from utils.noise import DiagonalGaussianDistribution
from torchvision.utils import make_grid
from torchvision.utils import save_image
import pytorch_lightning as pl
from scipy import integrate
# from losses.distribution_loss import Gaussian
# from torchvision.transforms import ToPILImage
from trainers.lit_ema import LitEma
from contextlib import contextmanager
from typing import Mapping, Any, List, Optional, Tuple, Union
import torch.nn.functional as F

# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint
import math

import numpy as np
import random

import torch.distributed as dist

import os

from piq import LPIPS


def logit_normal_sampling(t, m=0, s=1):
    return (1 / (s*np.sqrt(2*np.pi))) * (1 / (t*(1-t))) * torch.exp(-((torch.log2(t/(1-t)-m))**2/2*s**2))


class LitRectifiedFlow(pl.LightningModule):
    def __init__(
        self,
        data_config: Mapping[str, Any],
        rf_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        # loss_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any],
        compile: bool,
        sampler_type: str,
        sample_N: int,
        use_ema: bool = False,
        ):
        super().__init__()

        self.save_hyperparameters()
        
        autoencoder = instantiate_from_config(ae_config)
        
        # encoder
        self.encoder = autoencoder.encoder

        # decoder
        self.decoder = autoencoder.decoder

        self.t_learnable = rf_config['t_learnable']
        self.t_params = nn.ParameterDict({
            "LR2": nn.Parameter(torch.tensor(-0.693), requires_grad=self.t_learnable),
            "LR3": nn.Parameter(torch.tensor(0.693),  requires_grad=self.t_learnable),
        })

        self.ae_ckpt_path = ae_config['checkpoint']

        # instantiate model
        self.model = instantiate_from_config(model_config)
        if compile:
            self.model = torch.compile(self.model)
        # self.loss = instantiate_from_config(loss_config)
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.data_config = data_config

        # only for euler
        self.sample_N = sample_N
        print('Number of sampling steps:', self.sample_N)

        if sampler_type == 'rk45':
            self.sampler = self.rk45_sampler
        elif sampler_type == 'euler':
            self.sampler = self.euler_sampler
        elif sampler_type == 'dopri5':
            self.sampler = self.dopri5_sampler
        else:
            raise NotImplementedError()
        
        self.T = 1

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.train_resolutions = ["GT", "LR2", "LR3", "LR4"]
        self.interp_method = rf_config['interp_method']
        if self.interp_method == 'linear':
            self.interp_fn = linear_interpolate
        elif self.interp_method == 'cubic_spline':
            self.interp_fn = natural_cubic_spline_interpolate
        else:
            raise NotImplementedError()
        
        self.loss_type = rf_config['loss_type']
        if self.loss_type == 'l2':
            self.loss = self.l2_loss
        else:
            raise NotImplementedError()

        self.lpips = LPIPS(replace_pooling=True, reduction="none")
        self.lpips_scale = rf_config['lpips_scale']
        self.lpips_weighting = rf_config['lpips_weighting']
        
        self.test_y_channel = rf_config['test_y_channel']

        self.resolutions = ["LR2", "LR3", "LR4"]
        self.psnr_metrics = torch.nn.ModuleDict({
            res: torchmetrics.MeanMetric() for res in self.resolutions
        })
        self.nfe_metric = torchmetrics.MeanMetric()

        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')


        
    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None and self.trainer.logger is not None and hasattr(self.trainer.logger, "version"):
            self.version = self.trainer.logger.version
        else:
            self.version = "temp"
        
        autoencoder_ckpt = torch.load(self.ae_ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(autoencoder_ckpt["encoder"])
        self.decoder.load_state_dict(autoencoder_ckpt["decoder"])
        self.t_params.load_state_dict(autoencoder_ckpt["t_params"])

        for module in (self.encoder, self.decoder):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


    def configure_optimizers(self):
        optim_params = [{'params': self.model.parameters()},]

        if self.t_learnable:
            optim_params.append({'params': self.t_params.parameters()})

        optimizer = instantiate_from_config_with_arg(self.optimizer_config, optim_params)

        learning_rate_scheduler = instantiate_from_config_with_arg(
            self.scheduler_config, optimizer)
        
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": learning_rate_scheduler,
                    "interval": 'step',
                    "frequency": 1,},}

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def forward(self, x, sampler_type='euler', sample_N=None, reverse=False, set_grad=False):
        if sampler_type == 'rk45':
            x, nfev = self.rk45_sampler(x, reverse_t=reverse)
        elif sampler_type == 'euler':
            x, nfev = self.euler_sampler(x, reverse_t=reverse, sample_N=sample_N, set_grad=set_grad)
        else:
            raise NotImplementedError()

        return x, nfev
    
    
    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1
        
    @property
    def t_positions(self):
        t_GT = torch.tensor(0.0, device=self.device, dtype=self.t_params["LR2"].dtype)
        t_LR2  = torch.sigmoid(self.t_params["LR2"])
        t_LR3  = torch.sigmoid(self.t_params["LR3"])
        t_LR4  = torch.tensor(1.0, device=self.device, dtype=self.t_params["LR2"].dtype)
        return torch.stack([t_GT, t_LR2, t_LR3, t_LR4], dim=0)


    def on_train_batch_start(self, batch, batch_idx):
        x = batch['GT']
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()
    
    def training_step(self, batch, batch_idx):
        z_list = []
        x_list = []
        feature_hr = None

        for res in self.train_resolutions:
            img = batch[res]

            ### hr_skip ###
            z, feature = self.encoder(img)
            if res == 'GT':
                feature_hr = feature
            # z = self.encoder(img)
            ### hr_skip ###

            z_list.append(z)
            x_list.append(img)
        
        z_tensor = torch.stack(z_list, dim=0)
        x_tensor = torch.stack(x_list, dim=0)

        loss = self.loss_fn(
            latents=z_tensor, 
            imgs=x_tensor, 
            feature_hr = feature_hr, 
        )

        return loss

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)
    
    
    def l2_loss(self, x, y):
        return torch.mean((x - y) ** 2, dim=tuple(range(1, x.dim())))

    def loss_fn(self, latents, imgs, feature_hr, eps=1e-3):
        t_position = self.t_positions
        t_lr2 = t_position[1].item()
        t_lr3 = t_position[2].item()

        t = sample_t_uniform(latents[0].shape[0]).to(latents[0].device)


        ### Train Only [1, 2, 4] ###
        mask = torch.ones(t_position.size(0), 
                        dtype=torch.bool, 
                        device=latents.device)
        mask[2] = False

        latents    = latents[mask]      # -> shape (3, B, C, H, W)
        imgs       = imgs[mask]         # -> shape (3, B, C, H, W)
        t_position = t_position[mask] 
        ### Train Only [1, 2, 4] ###


        interp, deriv, second_deriv, third_deriv = self.interp_fn(latents, t, t_position)

        pred = self.model(interp, t * 999)

        loss_flow = self.loss(deriv, pred)
        total_loss = loss_flow
        logs = {'flow:': loss_flow.mean()}

        if self.lpips_scale > 0:
            idx = torch.searchsorted(t_position, t, right=True)
            idx = torch.clamp(idx, max=t_position.shape[0] - 1)
            t_right = t_position[idx]
            t_left = t_position[idx-1]
            delta_t = (t_right - t).view(-1, 1, 1, 1)

            if self.interp_method == 'linear':
                pred_z = interp + delta_t * pred
            elif self.interp_method == 'cubic_spline':
                # 3rd-order Taylor approximation
                pred_z = (
                    interp
                    + delta_t * pred
                    + 0.5 * (delta_t ** 2) * second_deriv
                    + (1.0 / 6.0) * (delta_t ** 3) * third_deriv
                )
            else:
                raise ValueError(f"Unknown interp_method: {self.interp_method}")

            ###
            pred_x = self.decoder(pred_z, feature_hr)
            # pred_x = self.decoder(pred_z)
            ###

            batch_indices = torch.arange(imgs.shape[1], device=imgs.device)
            x = imgs[idx, batch_indices, :, :, :]

            delta_t1 = (t_right - t).clamp(min=eps)
            seg_len = (t_right - t_left).clamp(min=eps)

            r = delta_t1 / seg_len

            lpips_weight = 1.0 / (r + eps) if self.lpips_weighting else 1.0
            
            loss_lpips = self.lpips(pred_x * 0.5 + 0.5, x * 0.5 + 0.5) * lpips_weight
            logs['lpips'] = loss_lpips.mean()
            total_loss += self.lpips_scale * loss_lpips

        loss = torch.mean(total_loss)

        self.log_dict(logs, prog_bar=True)

        return loss
    
    

    def on_validation_epoch_start(self):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)

    def validation_step(self, batch, batch_idx):
        img = batch["GT"]

        ### hr_skip ###
        z, feature_hr = self.encoder(img)
        # z = self.encoder(img)
        ### hr_skip ###


        t_eval = self.t_positions[1:].detach().cpu().numpy()

        pred, nfev_forward = self.sampler(z, t_eval=t_eval)

        global_indices = batch["img_idx"] + 1

        for resolution in self.resolutions:
            LR = batch[resolution]
            LR = (LR + 1) / 2

            z_LR_pred = pred[resolution]

            ### hr_skip ###
            LR_pred = self.decoder(z_LR_pred, feature_hr)
            # LR_pred = self.decoder(z_LR_pred)
            ### hr_skip ###

            LR_pred_clipped = torch.clamp(LR_pred, -1, 1)
            LR_pred_clipped = (LR_pred_clipped + 1) / 2

            psnr = calculate_psnr_pt(LR, LR_pred_clipped, crop_border=0, test_y_channel=self.test_y_channel)
            psnr = calculate_psnr_pt(LR.cpu(), LR_pred_clipped.cpu(), crop_border=0,
                                 test_y_channel=self.test_y_channel)

            self.psnr_metrics[resolution].update(psnr)


            save_dir = os.path.join("results", "rf", resolution)
            os.makedirs(save_dir, exist_ok=True)
            for i in range(LR_pred_clipped.shape[0]):
                sample_idx = global_indices[i] if isinstance(global_indices, list) else int(global_indices[i].item())
                save_path = os.path.join(save_dir, f"val_{sample_idx}_{resolution}.png")
                save_image(LR_pred_clipped[i], save_path)
        

        self.nfe_metric.update(nfev_forward)

    def on_validation_epoch_end(self):
        log_dict = {}
        for res in self.resolutions:
            log_dict[f'{res}_psnr'] = self.psnr_metrics[res].compute()
            self.psnr_metrics[res].reset()

        log_dict['nfe'] = self.nfe_metric.compute()
        self.nfe_metric.reset()

        self.log_dict(
            log_dict,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=True
        )

        if self.use_ema:
            self.model_ema.restore(self.model.parameters())


    @torch.no_grad()
    def rk45_sampler(self, z, reverse_t=False, t_eval=[1/3, 2/3, 1]):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
        model: A velocity model.
        z: If present, generate samples from latent code z.
        Returns:
        samples, number of function evaluations.
        """

        rtol=1e-5
        atol=1e-5
        method='RK45'
        eps=1e-3

        x = z

        def to_flattened_numpy(x):
            """Flatten a torch tensor x and convert it to numpy."""
            return x.detach().cpu().numpy().reshape((-1,))

        def from_flattened_numpy(x, shape):
            """Form a torch tensor with the given shape from a flattened numpy array x."""
            return torch.from_numpy(x.reshape(shape))


        def ode_func(t, x):
            x = from_flattened_numpy(x, z.shape).to(z.device).type(torch.float32)
            vec_t = torch.ones(x.shape[0], device=x.device) * t
            drift = self.model(x, vec_t * 999)

            return to_flattened_numpy(drift)

        # Black-box ODE solver for the probability flow ODE
        if reverse_t:
            t_span = (self.T-eps, 0.)
        else:
            t_span = (eps, self.T)
        solution = integrate.solve_ivp(ode_func, t_span, to_flattened_numpy(x),
                                        rtol=rtol, atol=atol, method=method, t_eval=t_eval, vectorized=True)
        nfe = solution.nfev

        pred = {}
        pred["GT"] = z
        pred["LR2"] = torch.tensor(solution.y[:, 0]).reshape(z.shape).to(z.device).type(torch.float32)
        pred["LR3"] = torch.tensor(solution.y[:, 1]).reshape(z.shape).to(z.device).type(torch.float32)
        pred["LR4"] = torch.tensor(solution.y[:, 2]).reshape(z.shape).to(z.device).type(torch.float32)

        return pred, nfe
    
