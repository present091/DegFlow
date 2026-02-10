import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import save_image
import numpy as np
from utils.common import instantiate_from_config
from scipy import integrate
from utils.interpolation import natural_cubic_spline_interpolate, linear_interpolate



class LitGenerator(pl.LightningModule):
    def __init__(
            self,
            version,
            root_dir,
            camera_type,
            ae_config,
            rf_config=None,
            mode='flow'           #'ae_interp'
        ):
        super().__init__()
        self.save_hyperparameters()
        self.version     = version
        self.root_dir    = root_dir
        self.camera_type = camera_type
        self.mode        = mode

        # Autoencoder
        autoencoder = instantiate_from_config(ae_config)
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.ae_ckpt_path = ae_config['checkpoint']

        # Flow-model
        if self.mode == 'flow':
            assert rf_config is not None, "needs rf_config."
            self.flow_model = instantiate_from_config(rf_config)
            self.rf_ckpt_path = rf_config['checkpoint']
        
        self.total_time = 0
        self.n = 0

    def setup(self, stage):
        ae_ckpt = torch.load(self.ae_ckpt_path, map_location=self.device)
        self.encoder.load_state_dict(ae_ckpt["encoder"])
        self.decoder.load_state_dict(ae_ckpt["decoder"])
        self.encoder.eval().to(memory_format=torch.channels_last)
        self.decoder.eval().to(memory_format=torch.channels_last)

        if self.mode == 'flow':
            rf_ckpt = torch.load(self.rf_ckpt_path, map_location=self.device)
            self.flow_model.load_state_dict(rf_ckpt['model'])
            self.flow_model.eval()

    @torch.no_grad()
    def rk45_sampler(self, z, t_eval, T=1):
        rtol, atol = 1e-5, 1e-5
        method, eps = 'RK45', 1e-3
        def to_flat(x): return x.detach().cpu().numpy().reshape(-1)
        def from_flat(x, shape): return torch.from_numpy(x.reshape(shape))

        def ode_func(t, x_flat):
            x = from_flat(x_flat, z.shape).to(z.device).float()
            vec_t = torch.full((x.size(0),), t, device=x.device)
            drift = self.flow_model(x, vec_t * 999)
            return to_flat(drift)

        sol = integrate.solve_ivp(
            ode_func, (eps, T), to_flat(z), rtol=rtol, atol=atol,
            method=method, t_eval=t_eval, vectorized=True
        )
        outputs = [
            torch.tensor(sol.y[:, i])
                 .reshape(z.shape).to(z.device).float()
            for i in range(len(sol.t))
        ]
        return outputs, sol.nfev

    def on_predict_epoch_start(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        raw = batch['prefix']
        if isinstance(raw, (list, tuple)): 
            raw = raw[0]
        prefix = (raw.decode() if isinstance(raw, (bytes, bytearray)) 
                else raw.item() if torch.is_tensor(raw) else raw)

        resolutions = np.linspace(1, 4, 31)
        out_base = os.path.join(self.root_dir, self.camera_type, f"version{self.version}")

        img_gt = batch['GT']           # [-1,1]
        hr     = (img_gt + 1) / 2      # [0,1]

        if self.mode == 'flow':
            z, feature = self.encoder(img_gt)

            t_eval = (resolutions - 1) / 3
            preds, _ = self.rk45_sampler(z, t_eval[1:])

            for scale, p in zip(resolutions[1:], preds):
                lr = self.decoder(p, feature)
                lr.clamp_(-1, 1)

                s_str  = f"{scale:.10f}".rstrip('0').rstrip('.')
                out_dir = os.path.join(out_base, s_str)
                os.makedirs(out_dir, exist_ok=True)

                save_image(hr[0], os.path.join(out_dir, f"{prefix}_HR.png"))

                save_image(((lr + 1) / 2)[0],
                           os.path.join(out_dir, f"{prefix}_LR{s_str}.png"))



        elif self.mode == 'ae_interp':
            z_hr, feature_hr = self.encoder(batch['GT'])
            latent_scales = {
                1.0: z_hr,
                2.0: self.encoder(batch['LR2'])[0],
                4.0: self.encoder(batch['LR4'])[0]
            }

            # control_points shape: (3, B, C, H, W)
            control_points = torch.stack([
                latent_scales[1.0],
                latent_scales[2.0],
                latent_scales[4.0]
            ], dim=0)
            control_positions = torch.tensor([0.0, 1/3, 1.0],
                                             device=control_points.device)

            B = control_points.shape[1]
            t_values = (resolutions - 1) / 3

            for scale, t_val in zip(resolutions[1:], t_values[1:]):
                t_tensor = torch.full((B,), float(t_val),
                                      device=control_points.device)
                z_s, deriv, second_deriv, third_deriv = natural_cubic_spline_interpolate(
                    control_points, t_tensor, control_positions
                )
                # z_s, deriv, second_deriv, third_deriv = linear_interpolate(
                #     control_points, t_tensor, control_positions
                # )

                s_str  = f"{scale:.10f}".rstrip('0').rstrip('.')
                out_dir = os.path.join(out_base, s_str)
                os.makedirs(out_dir, exist_ok=True)

                save_image(hr[0], os.path.join(out_dir, f"{prefix}_HR.png"))

                ###
                # lr = torch.clamp(self.decoder(z_s), -1, 1)
                lr = torch.clamp(self.decoder(z_s, feature_hr), -1, 1)
                ###

                save_image(((lr + 1) / 2)[0],
                           os.path.join(out_dir, f"{prefix}_LR{s_str}.png"))


        else:
            raise ValueError(f"Unknown mode: {self.mode}")
