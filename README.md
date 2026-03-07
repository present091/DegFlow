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

For unseen intermediate degradation scales, we extrapolate an intermediate latent toward the next discrete degradation level using a third-order Taylor expansion:

$$
\hat{z}_{t_{k+1}} = z_t + \hat{z}'_t \Delta t + \frac{1}{2} z''_t \Delta t^2 + \frac{1}{6} z'''_t \Delta t^3
$$

We then decode the extrapolated latent and compute the LPIPS loss against the ground-truth LR image at the next degradation level:

$$
\mathcal{L}_{\mathrm{LPIPS}} = \mathrm{LPIPS}\left(I_{s_{k+1}}, D_\theta\left(\hat{z}_{t_{k+1}}\right)\right)
$$

This enables perceptual supervision at unseen intermediate degradation scales without requiring direct ground-truth LR images.

---

## Results

### Generated Continuous-Scale LR Images

<img width="60%" alt="image" src="https://github.com/user-attachments/assets/6774d968-5496-49a4-956c-25bbb8ae7cf4" />

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


