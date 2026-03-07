<div align="center">
<h1>Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution [AAAI 2026]</h1>

<h4>
  Hyeonjae Kim*,
  <a href="https://dongjinkim9.github.io">Dongjin Kim</a>*,
  Eugene Jin,
  <a href="https://sites.google.com/view/lliger9/team/taehyunkim">Tae Hyun Kim</a><sup>&#8224;</sup>
</h4>

<b><sub><sup>* Equal contribution.  <sup>&#8224;</sup> Corresponding author.</sup></sub></b>

[![arXiv](https://img.shields.io/badge/Arxiv-📄Paper-8A2BE2)](https://arxiv.org/abs/2602.04193)
&nbsp;
[![Project_page](https://img.shields.io/badge/Project-📖Page-8A2BE2)]()
&nbsp;

</div>

---


## Overview

This repository contains the official implementation of **Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution**.

<p align="center">
  <img width="60%" alt="image" src="https://github.com/user-attachments/assets/a8f4571a-acc4-47d8-a4d6-93e0081a43bf" />
</p>

We propose **DegFlow**, a continuous degradation modeling framework that can **synthesize realistic LR images at continuous scales from a single HR input at inference time** by learning degradation trajectories in a constrained latent space with Latent Flow Matching (LFM). The generated continuous-scale LR images provide realistic supervision beyond limited discrete real HR–LR pairs, leading to improved real-world arbitrary-scale super-resolution (ASSR) performance.

<p align="center">
  <img width="1437" height="780" alt="image" src="https://github.com/user-attachments/assets/d5b03b58-04f3-4580-ac3a-0f38cc4cea3c" />
</p>

### LPIPS-based Perceptual Supervision for Nonlinear Flow Matching

<p align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/e4c51b80-11be-45ae-838a-a8af3115ef96" />
</p>
<p align="center">
  <sub>Illustration of applying LPIPS to nonlinear flow matching for perceptually meaningful degradation modeling.</sub>
</p>

For **unseen intermediate degradation scales**, we extrapolate an intermediate latent toward the next discrete degradation level using a **third-order Taylor expansion**:

$$
\hat{z}_{t_{k+1}} = z_t + \hat{z}'_t \Delta t + \frac{1}{2} z''_t \Delta t^2 + \frac{1}{6} z'''_t \Delta t^3
$$

We then decode the extrapolated latent and compute the **LPIPS loss** against the ground-truth LR image at the next degradation level:

$$
\mathcal{L}_{\mathrm{LPIPS}} = \mathrm{LPIPS}\left(I_{s_{k+1}}, D_\theta\left(\hat{z}_{t_{k+1}}\right)\right)
$$

This enables **perceptual supervision** even at **unseen intermediate degradation scales**, without requiring direct ground-truth LR images.

---

## Results

### Generated Continuous-Scale LR Images

<img width="60%" alt="image" src="https://github.com/user-attachments/assets/6774d968-5496-49a4-956c-25bbb8ae7cf4" />

### Quantitative Results on Fixed-Scale SR

**Table 2.** Fixed-scale SR results on the RealSR ×3 test set.  
**Best** results are in **bold**, and *second-best* results are in *italic*.

| Model | SR Train Set | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------------|-------:|-------:|--------:|
| RCAN | RealSR ×3 | *30.68* | *0.8641* | 0.3243 |
| RCAN | RealSR ×2, ×4 | 30.30 | 0.8596 | 0.3281 |
| RCAN | InterFlow ×2~×4 | 30.57 | 0.8631 | **0.3155** |
| RCAN | Ours ×2~×4 | **30.72** | **0.8650** | *0.3221* |
| HAN | RealSR ×3 | *30.76* | *0.8659* | 0.3216 |
| HAN | RealSR ×2, ×4 | 30.43 | 0.8616 | 0.3261 |
| HAN | InterFlow ×2~×4 | 30.68 | 0.8644 | **0.3167** |
| HAN | Ours ×2~×4 | **30.82** | **0.8660** | *0.3212* |
| SwinIR | RealSR ×3 | *30.69* | *0.8647* | 0.3217 |
| SwinIR | RealSR ×2, ×4 | 30.23 | 0.8597 | 0.3255 |
| SwinIR | InterFlow ×2~×4 | 30.56 | 0.8634 | **0.3166** |
| SwinIR | Ours ×2~×4 | **30.78** | **0.8658** | *0.3193* |
| HAT | RealSR ×3 | *30.71* | *0.8645* | 0.3221 |
| HAT | RealSR ×2, ×4 | 30.39 | 0.8607 | 0.3248 |
| HAT | InterFlow ×2~×4 | 30.65 | *0.8645* | **0.3135** |
| HAT | Ours ×2~×4 | **30.86** | **0.8668** | *0.3186* |
| MambaIR | RealSR ×3 | *30.62* | 0.8636 | 0.3208 |
| MambaIR | RealSR ×2, ×4 | 30.29 | *0.8660* | 0.3240 |
| MambaIR | InterFlow ×2~×4 | 30.51 | 0.8625 | **0.3138** |
| MambaIR | Ours ×2~×4 | **30.73** | **0.8686** | *0.3152* |

### Quantitative Results on Arbitrary-Scale SR

**Table 3.** Arbitrary-scale SR results on the RealSR ×3 and RealArbiSR test sets.  
**Best** results are in **bold**, and *second-best* results are in *italic*.

| Model | Train Set | LR Generation | ×3 PSNR | ×3 SSIM | ×3 LPIPS | ×2.5 PSNR | ×2.5 SSIM | ×2.5 LPIPS | ×3 PSNR (RealArbiSR) | ×3 SSIM | ×3 LPIPS | ×3.5 PSNR | ×3.5 SSIM | ×3.5 LPIPS |
|------|-----------|---------------|--------:|--------:|---------:|----------:|----------:|-----------:|---------------------:|--------:|---------:|----------:|----------:|-----------:|
| MetaSR | RealSR ×3 | None (Oracle) | 30.43 | 0.8572 | 0.3311 | 29.65 | 0.8679 | 0.3330 | 29.58 | 0.8338 | 0.3557 | 28.15 | 0.7974 | 0.3996 |
| MetaSR | RealSR ×1 | Bicubic (Baseline) | 28.99 | 0.8165 | 0.3488 | 30.05 | 0.8473 | 0.3087 | 28.77 | 0.8042 | 0.3544 | 27.86 | 0.7711 | 0.3886 |
| MetaSR | RealSR ×1 | BSRGAN | 28.15 | 0.8114 | 0.3867 | 28.41 | 0.8326 | 0.3679 | 27.41 | 0.7932 | 0.3971 | 26.76 | 0.7625 | 0.4199 |
| MetaSR | RealSR ×1 | Real-ESRGAN | 26.90 | 0.8077 | 0.3813 | 27.32 | 0.8177 | 0.3738 | 26.50 | 0.7821 | 0.4014 | 25.94 | 0.7305 | 0.4380 |
| MetaSR | RealSR ×2, ×4 | InterFlow | *30.42* | **0.8569** | *0.3222* | *30.71* | *0.8703* | *0.3099* | 29.40 | 0.8302 | 0.3504 | 28.50 | 0.7983 | 0.3828 |
| MetaSR | RealSR ×2, ×4 | Ours | **30.58** | *0.8565* | **0.3190** | **30.88** | **0.8713** | **0.2995** | **29.63** | **0.8321** | **0.3429** | **28.71** | **0.8008** | **0.3780** |
| LIIF | RealSR ×3 | None (Oracle) | 30.43 | 0.8578 | 0.3324 | 30.71 | 0.8718 | 0.3222 | 29.56 | 0.8336 | 0.3579 | 28.66 | *0.8028* | 0.3861 |
| LIIF | RealSR ×1 | Bicubic (Baseline) | 29.00 | 0.8167 | 0.3290 | 30.04 | 0.8472 | **0.3096** | 28.76 | 0.8042 | 0.3550 | 27.86 | 0.7711 | 0.3892 |
| LIIF | RealSR ×1 | BSRGAN | 28.23 | 0.8133 | 0.3875 | 28.28 | 0.8303 | 0.3719 | 27.32 | 0.7912 | 0.4009 | 26.75 | 0.7617 | 0.4242 |
| LIIF | RealSR ×1 | Real-ESRGAN | 27.07 | 0.8090 | 0.3817 | 27.24 | 0.8135 | 0.3755 | 26.36 | 0.7777 | 0.4025 | 25.73 | 0.7492 | 0.4243 |
| LIIF | RealSR ×2, ×4 | InterFlow | *30.44* | **0.8581** | *0.3263* | *30.70* | *0.8705* | *0.3144* | 29.38 | 0.8307 | 0.3547 | 28.44 | 0.7985 | 0.3860 |
| LIIF | RealSR ×2, ×4 | Ours | **30.61** | *0.8577* | **0.3251** | **30.99** | **0.8729** | **0.3105** | **29.74** | **0.8341** | **0.3517** | **28.78** | **0.8027** | **0.3845** |
| CiaoSR | RealSR ×3 | None (Oracle) | *30.65* | **0.8609** | 0.3251 | 30.61 | 0.8705 | 0.3105 | *29.59* | **0.8339** | 0.3487 | 28.54 | *0.8011* | 0.3810 |
| CiaoSR | RealSR ×1 | Bicubic (Baseline) | 28.98 | 0.8160 | 0.3496 | 30.04 | 0.8472 | 0.3079 | 28.76 | 0.8037 | 0.3545 | 27.85 | 0.7708 | 0.3881 |
| CiaoSR | RealSR ×1 | BSRGAN | 28.55 | 0.8288 | 0.3638 | 28.88 | 0.8443 | 0.3490 | 27.90 | 0.8046 | 0.3797 | 27.29 | 0.7755 | 0.4069 |
| CiaoSR | RealSR ×1 | Real-ESRGAN | 27.48 | 0.8200 | 0.3696 | 27.86 | 0.8341 | 0.3539 | 26.95 | 0.7953 | 0.3823 | 26.31 | 0.7664 | 0.4091 |
| CiaoSR | RealSR ×2, ×4 | InterFlow | *30.52* | **0.8590** | *0.3162* | *30.69* | *0.8702* | **0.3057** | 29.36 | 0.8298 | *0.3408* | 28.52 | 0.8000 | *0.3795* |
| CiaoSR | RealSR ×2, ×4 | Ours | **30.70** | **0.8590** | **0.3153** | **31.03** | **0.8739** | *0.3059* | **29.58** | *0.8318* | 0.3439 | **28.77** | **0.8032** | **0.3793** |

### Unseen-Scale SR Results

<img width="1297" height="432" alt="image" src="https://github.com/user-attachments/assets/d2a65c08-9f75-4bb6-8b0a-9ff4095dbc40" />



---

## Dataset Preparation

We use the **RealSR Version 2** dataset for both training and evaluation.

To be specific, we use a dataset reconstructed from **InterFlow**
(*Learning Controllable Degradation for Real-World Super-Resolution via Constrained Flows*),
containing only overlapping HR–LR pairs.

You may refer to the official
[InterFlow repository](https://github.com/dongjinkim9/InterFlow/blob/main/interflow/README.md#Training),
where the dataset is available for download.


After downloading, set the dataset root path in the config files:

`configs/datasets/realsr_train.yaml`

`configs/datasets/realsr_train_allscale.yaml`

`configs/datasets/realsr_test.yaml`

`configs/datasets/realsr_test_allscale.yaml`

```yaml
dataset:
  params:
    dataroot: '/path/to/RealSR_v2_ordered/'
```

We assume an ordered dataset structure such as `RealSR_v2_ordered`.
Please modify the dataset loader if your directory structure differs.

---

## Training

Training consists of two stages.

### Stage 1 — Train RAE (Residual Autoencoder)

```bash
python main.py --config configs/train_lit_ae.yaml
```

This stage trains the latent autoencoder to construct a constrained latent space.

---

### Stage 2 — Train LFM (Latent Flow Matching)

```bash
python main.py --config configs/train_lit_rf.yaml
```

This stage trains the latent flow model in the learned latent space.

---

## Dataset Generation

To generate arbitrary-scale real LR samples or latent trajectories, configure:

`configs/generate.yaml`

Set pretrained checkpoint paths:

```yaml
model:
  params:
    ae_config:
      checkpoint: /path/to/autoencoder_checkpoint.ckpt
    rf_config:
      checkpoint: /path/to/flow_model_checkpoint.ckpt
```

Run generation:

```bash
python generate.py --config configs/generate.yaml
```

### Generation with External HR-Only Datasets

Furthermore, DegFlow generation can also be performed using external datasets that contain **HR images only**.

For example, **DIV2K** is currently supported.
You can enable generation by configuring the `data` section as follows:

```yaml
data:
  target: datasets.data_module.GenerationDataModule
  params:
    dataset:
      target: datasets.dataset.DIV2K_HR_only_dataset
      params:
        root_dir: '/path/to/DIV2K dataset'
```

You can also perform generation with any custom dataset that contains HR images only by implementing or adapting a compatible dataset class.


---

## Pretrained Checkpoints

We provide pretrained checkpoints for both the autoencoder and flow models.

### Residual Autoencoder

| Model       | Download                  |
| ----------- | ------------------------- |
| Autoencoder | [Google Drive](https://drive.google.com/file/d/1qOAa5FwGX9fAurpB3hJtcIfrARBkTjBJ/view?usp=drive_link) |

---

### Flow Models

| Model      | Download                  |
| ---------- |-------------------------- |
| Flow Model | [Google Drive](https://drive.google.com/file/d/1t-V-hiJOP5g1LI-qU3VCArR6bM5OABEP/view?usp=drive_link) |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026degflow,
  title     = {Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution},
  author    = {Kim, Hyeonjae and Kim, Dongjin and Jin, Eugene and Kim, Taehyun},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```
---

## Acknowledgements

This project builds upon ideas from the following works:

- **Learning Controllable Degradation for Real-World Super-Resolution via Constrained Flows** (ICML 2023)
- **Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow** (ICLR 2023)
- **Improving the Training of Rectified Flows** (NeurIPS 2024)

We sincerely thank the authors for their inspiring and foundational contributions.


