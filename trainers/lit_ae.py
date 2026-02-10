import pytorch_lightning as pl
from torch import nn
import torch

from typing import Mapping, Any, List, Optional, Tuple, Union
import numpy as np
from utils.common import instantiate_from_config, instantiate_from_config_with_arg
from utils.metrics import calculate_psnr_pt
import torch.distributed as dist
import torch.nn.functional as F
from torchmetrics import MetricCollection, MeanMetric

import torch.nn.functional as F

from piq import LPIPS



class LitAE(pl.LightningModule):
    def __init__(
        self, 
        misc_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        ae_config: Mapping[str, Any],
        model_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any] = None,
        ):
        super().__init__()

        self.save_hyperparameters()

        self.misc_config = misc_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.lpips_loss_scale = ae_config['lpips_loss_scale']
        if self.lpips_loss_scale > 0:
            self.lpips = LPIPS(replace_pooling=True, reduction="none")

        self.test_y_channel = ae_config['test_y_channel']

        autoencoder = instantiate_from_config(model_config)

        if self.misc_config.compile:
            self.autoencoder = torch.compile(autoencoder)

        self.t_params = nn.ParameterDict({
            "LR2": nn.Parameter(torch.tensor(-0.693), requires_grad=ae_config['t_learnable']),
            "LR3": nn.Parameter(torch.tensor(0.693), requires_grad=ae_config['t_learnable'])
        })

        self.training_step = self.training_random_res

        self.val_recon_psnr = nn.ModuleDict({
            res: MeanMetric(dist_sync_on_step=True)
            for res in ["GT", "LR2", "LR3", "LR4"]
        })


    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None and self.trainer.logger is not None and hasattr(self.trainer.logger, "version"):
            self.version = self.trainer.logger.version
        else:
            self.version = "temp"

    def configure_optimizers(self):
        optimizer = instantiate_from_config_with_arg(
            self.optimizer_config,
            [
                {'params': self.autoencoder.parameters()},

            ]
        )

        optim_config = {"optimizer": optimizer}

        if self.scheduler_config:
            optim_config["lr_scheduler"] = {
                "scheduler": instantiate_from_config_with_arg(
                    self.scheduler_config, optimizer
                ),
                "interval": 'step',
                "frequency": 1,
            }


        return optim_config
    
    def reconstruction_loss(self, x_reconstructed, x):
        return F.mse_loss(x_reconstructed, x, reduction='mean')

    def get_world_size(self):
        if dist.is_initialized():
            return dist.get_world_size()
        else:
            return 1

    def on_train_batch_start(self, batch, batch_idx):
        x = batch['GT']
        self.global_batch_size = int(x.shape[0]) * self.get_world_size()

    
    def training_random_res(self, batch, batch_idx):
        hr = batch['GT']
        x = batch['LQ']

        z_hr, feature_hr = self.autoencoder.encoder(hr)

        z, feature = self.autoencoder.encoder(x)
        x_rec = self.autoencoder.decoder(z, feature_hr)
        loss_recon = self.reconstruction_loss(x_rec, x)
        logs = {'recon:': loss_recon}

        loss = loss_recon

        if self.lpips_loss_scale > 0:
            loss_lpips = self.lpips(x_rec * 0.5 + 0.5, x * 0.5 + 0.5).mean()
            logs['lpips'] = loss_lpips.mean()
            loss += self.lpips_loss_scale * loss_lpips

        self.log_dict(logs, prog_bar=True)

        return loss


    def on_before_optimizer_step(self, optimizer):
        warmup_iter = self.misc_config.warmup
        if warmup_iter is None or warmup_iter <= 0:
            return

        if self.trainer.global_step < warmup_iter:
            base_lr = self.optimizer_config.params.lr
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / warmup_iter)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * base_lr


    def validation_step(self, batch, batch_idx):
        resolutions = ["GT", "LR2", "LR3", "LR4"]
        latent_dict = {}

        z_hr, feature_hr = self.autoencoder.encoder(batch['GT'])

        for res in resolutions:
            img = batch[res]

            ###
            latent, feature = self.autoencoder.encoder(img); latent_dict[res] = latent
            recon = torch.clamp(self.autoencoder.decoder(latent, feature_hr), -1, 1).add(1).div(2)
            # latent = self.autoencoder.encoder(img); latent_dict[res] = latent
            # recon = torch.clamp(self.autoencoder.decoder(latent), -1, 1).add(1).div(2)
            ###

            img_norm = img.add(1).div(2)

            psnr_val = calculate_psnr_pt(
                img_norm, recon, crop_border=0, test_y_channel=self.test_y_channel
            )
            self.val_recon_psnr[res].update(psnr_val)

    def on_validation_epoch_end(self):
        logs = {}
        for res, metric in self.val_recon_psnr.items():
            logs[res] = metric.compute()
            metric.reset()

        logs['t2'] = torch.sigmoid(self.t_params["LR2"]).detach()
        logs['t3'] = torch.sigmoid(self.t_params["LR3"]).detach()

        self.log_dict(
            logs,
            prog_bar=True,
            sync_dist=True,
            rank_zero_only=True,
            on_epoch=True
        )
