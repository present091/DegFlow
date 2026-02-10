import torch
from torch.utils.data import DataLoader, DistributedSampler

from omegaconf import OmegaConf

from argparse import ArgumentParser

from utils.common import instantiate_from_config

import pytorch_lightning as pl



def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    model = instantiate_from_config(config.model)
    data_module = instantiate_from_config(config.data)


    trainer = pl.Trainer(**config.lightning.trainer)

    trainer.predict(model, datamodule=data_module)


if __name__ == '__main__':
    main()
