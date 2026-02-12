# Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution

**Hyeonjae Kim***, **Dongjin Kim***, Eugene Jin, Taehyun Kim
*Equal Contribution

[📄 [AAAI 2026] Paper (PDF)](https://arxiv.org/pdf/2602.04193.pdf)


---


## Overview

This repository contains the official implementation of **Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution**.

<img width="778" height="647" alt="image" src="https://github.com/user-attachments/assets/a8f4571a-acc4-47d8-a4d6-93e0081a43bf" />


We propose a framework that models continuous real-world degradation trajectories in a constrained latent space using **Latent Flow Matching (LFM)**.
Our method enables arbitrary-scale real-world super-resolution by learning physically meaningful degradation paths between HR and LR images.

---


## Dataset

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

After downloading, specify checkpoint paths in config files:

```yaml
checkpoint: /path/to/checkpoint.ckpt
```

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

