# Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution [AAAI 2026]

**Hyeonjae Kim***, **Dongjin Kim***, Eugene Jin, Taehyun Kim
*Equal Contribution

[📄 [AAAI 2026] Paper (PDF)](https://arxiv.org/pdf/2602.04193.pdf)


---


## Overview

This repository contains the official implementation of **Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution**.

<p align="center">
  <img width="778" height="647" alt="image" src="https://github.com/user-attachments/assets/a8f4571a-acc4-47d8-a4d6-93e0081a43bf" />
</p>

We propose **DegFlow**, a continuous degradation modeling framework that can **synthesize realistic LR images at continuous scales from a single HR input at inference time** by learning degradation trajectories in a constrained latent space with Latent Flow Matching (LFM). The generated continuous-scale LR images provide realistic supervision beyond limited discrete real HR–LR pairs, leading to improved real-world arbitrary-scale super-resolution (ASSR) performance.


---

## Results

### Generated Continuous-Scale LR Images

<img width="60%" alt="image" src="https://github.com/user-attachments/assets/6774d968-5496-49a4-956c-25bbb8ae7cf4" />

### Unseen-Scale SR Results

<img width="1297" height="432" alt="image" src="https://github.com/user-attachments/assets/d2a65c08-9f75-4bb6-8b0a-9ff4095dbc40" />


---

## Dataset Preparation

We use the **RealSR Version 2** dataset for both training and evaluation.

Download the dataset from:
https://github.com/csjcai/RealSR

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

---

## Pretrained Checkpoints

Pretrained models will be released soon.

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

