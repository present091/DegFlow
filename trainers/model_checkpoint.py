import os
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

class AutoencoderCheckpoint(ModelCheckpoint):
    def __init__(self, save_dir: str = "./checkpoints/autoencoder/", save_interval: int = 10000, **kwargs):
        super().__init__(dirpath=save_dir, every_n_train_steps=save_interval, save_top_k=-1, **kwargs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        version = getattr(pl_module, "version", "temp")
        global_step = trainer.global_step
        filename = f"autoencoder-step{global_step}-version{version}.pth"
        save_path = os.path.join(self.save_dir, filename)

        checkpoint_dict = {
            "encoder": pl_module.autoencoder.encoder.state_dict(),
            "decoder": pl_module.autoencoder.decoder.state_dict(),
            "t_params": pl_module.t_params.state_dict(),
        }
        torch.save(checkpoint_dict, save_path)
        print(f"Checkpoint saved at {save_path}")


class RectifiedFlowCheckpoint(ModelCheckpoint):
    def __init__(self, save_dir: str= "./checkpoints/rectified_flow/", save_interval: int = 10000, **kwargs):
        super().__init__(dirpath=save_dir, every_n_train_steps=save_interval, save_top_k=-1, **kwargs)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        version = getattr(pl_module, "version", "temp")
        global_step = trainer.global_step
        filename = f"rectified_flow-step{global_step}-version{version}.pth"
        save_path = os.path.join(self.save_dir, filename)

        if getattr(pl_module, "use_ema", False):
            with pl_module.ema_scope(context="checkpoint"):
                model_state = pl_module.model.state_dict()
        else:
            model_state = pl_module.model.state_dict()

        checkpoint_dict = {
            "model": model_state,
        }
        torch.save(checkpoint_dict, save_path)
        print(f"Checkpoint saved at {save_path}")
