<div align="center">

# Temporal Context Consistency Above All:<br>Enhancing Long-Term Anticipation by Learning and Enforcing Temporal Constraints

[**Alberto Maté-Angulo**](https://github.com/alberto-mate) · [**Mariella Dimiccoli**](https://www.iri.upc.edu/staff/mdimiccoli)

*Institut de Robòtica i Informàtica Industrial (CSIC-UPC), Barcelona, Spain*

[![arXiv](https://img.shields.io/badge/arXiv-2412.19424-b31b1b.svg)](https://arxiv.org/abs/2412.19424)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Overview

Official PyTorch implementation and pretrained models for **TCCA** (*Temporal Context Consistency Above All: Enhancing Long-Term Anticipation by Learning and Enforcing Temporal Constraints*).

Long-term action anticipation (LTA) aims to predict future action labels and their durations from an initial untrimmed video observation. Standard parallel query-based decoders predict future segments simultaneously without explicitly modeling their temporal dependencies, often resulting in locally inconsistent transitions and globally implausible trajectories. **TCCA** addresses this challenge at two complementary levels:

1. **Bi-Directional Action Context Regularizer (BACR):** Enforces local temporal coherence between adjacent predicted segments by training each query representation to forecast both its preceding ($\hat{a}_{\text{past}}$) and succeeding ($\hat{a}_{\text{fut}}$) action distributions via bidirectional KL-divergence alignment, while conditioning duration estimation jointly on query features and predicted action probabilities.
2. **Conditional Random Field (CRF) Sequence Layer:** Explicitly learns an action transition matrix over the full prediction horizon to perform global sequence optimization via Viterbi decoding (`decode`), Exact $K$-Best Viterbi (`decode_kbest`), and stochastic sequence sampling via Forward-Filtering Backward-Sampling (`decode_stochastic_ffbs`).
3. **Hierarchical Observation Segmentation:** Leverages a multi-stage temporal action segmentation encoder (`LTContext` with truncated MSE smoothing loss) to extract structured past action representations and ground the first future query in the final observed action state.

<p align="center">
  <img src="docs/TCCA.png" alt="Overview of the TCCA Architecture" width="92%"/>
</p>

---

## Main Results & Pretrained Checkpoints

All pretrained checkpoints for the four standard LTA benchmarks are provided directly in [`checkpoints_best/`](checkpoints_best/).

### Procedural Activity Benchmarks (Dense Long-Term Anticipation — mMoC %)

| Benchmark | Splits | Queries ($K$) | $\omega$ | $\alpha=0.2$ (0.1 / 0.2 / 0.3 / 0.5) | $\alpha=0.3$ (0.1 / 0.2 / 0.3 / 0.5) | **mMoC (%)** | Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Breakfast** | 4 | 8 | 0.1 | 28.55 / 25.81 / 24.90 / 24.20 | 36.26 / 32.95 / 32.16 / 28.24 | **29.13** | [`checkpoints_best/breakfast/`](checkpoints_best/breakfast/) |
| **50Salads** | 5 | 20 | 1.0 | 40.82 / 29.83 / 24.47 / 20.42 | 37.12 / 28.31 / 22.85 / 18.43 | **28.78** | [`checkpoints_best/50salads/`](checkpoints_best/50salads/) |

### Egocentric Activity Benchmarks (Multi-Label Long-Term Anticipation — mAP %)

| Benchmark | Queries ($K$) | $\omega$ | **All Classes mAP (%)** | **Many-Shot (Freq) mAP (%)** | **Rare-Shot mAP (%)** | Checkpoint Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EGTEA Gaze+** | 30 | 1.0 | **80.3** | **85.8** | **65.7** | [`checkpoints_best/EGTEA/`](checkpoints_best/EGTEA/) |
| **EPIC-Kitchens-55** | 250 | 1.0 | **40.6** | **59.4** | **30.7** | [`checkpoints_best/EPIC/`](checkpoints_best/EPIC/) |

---

## Installation

Clone this repository and create the Conda environment using the provided [`TCCAenv.yaml`](TCCAenv.yaml) specification:

```bash
git clone https://github.com/alberto-mate/TCCA.git
cd TCCA

conda env create -f TCCAenv.yaml
conda activate TCCAenv
```

---

## Dataset Preparation

Download the pre-extracted I3D features, ground-truth annotations, and split bundles for the four benchmarks:

- **Breakfast & 50Salads:** [Zenodo Download Link](https://zenodo.org/records/3625992#.Xiv9jGhKhPY)
- **EGTEA Gaze+ & EPIC-Kitchens-55:** [Google Drive Download Link](https://drive.google.com/file/d/1um-jqoJbnzVk8HG_ACuVcZ62pPCTJtVe/view?usp=sharing)

Organize the extracted files inside the `datasets/` directory at the root of the repository as follows:

```text
TCCA/
└── datasets/
    ├── breakfast/
    │   ├── features/
    │   ├── groundTruth/
    │   ├── splits/
    │   └── mapping.txt
    ├── 50salads/
    ├── EGTEA/
    └── EPIC/
```

---

## Evaluation (Pretrained Models)

Evaluation scripts for all four benchmarks are provided in the [`scripts/`](scripts/) folder:

### 1. Breakfast (Splits 1–4)
```bash
for split in 1 2 3 4; do
    bash scripts/bf_predict_best.sh $split
done
```

### 2. 50Salads (Splits 1–5)
```bash
for split in 1 2 3 4 5; do
    bash scripts/50s_predict_best.sh $split
done
```

### 3. EGTEA Gaze+ (Split 1)
```bash
bash scripts/egtea_predict_best.sh 1
```

### 4. EPIC-Kitchens-55 (Split 1)
```bash
bash scripts/epic_predict_best.sh 1
```

---

## Training

### Direct Command-Line Training
You can train **TCCA** directly via [`main.py`](main.py) on any benchmark and split. For example:

**Train on Breakfast (Split 1):**
```bash
python main.py --dataset breakfast --split 1 --seg --task long --anticipate \
    --model tcca_LTC --hidden_dim 64 --hidden_dim_decoder 128 \
    --n_encoder_layer 2 --n_decoder_layer 2 --n_query 8 --n_head 8 \
    --max_pos_len 2000 --sample_rate 3 --epochs 50 --batch_size 16 \
    --lr 1e-3 --weight_decay 0.005 --dropout 0.3 --clip_grad_norm 10 \
    --segmentation_loss cross_entropy+tmse --tsme_seg_weight 0.2 \
    --duration action_partially_dependent \
    --anticipate_next --anticipate_prev --crf --crf_transition_weight 0.1 \
    --ltc_long_term_attn_g 16 --ltc_num_stages 4
```

**Train on 50Salads (Split 1):**
```bash
python main.py --dataset 50salads --split 1 --seg --task long --anticipate \
    --model tcca_LTC --hidden_dim 128 --hidden_dim_decoder 256 \
    --n_decoder_layer 3 --n_query 20 --max_pos_len 3100 --sample_rate 3 \
    --epochs 50 --batch_size 8 --lr 1e-3 --weight_decay 0.005 --dropout 0.3 \
    --segmentation_loss cross_entropy+tmse --tsme_seg_weight 0.2 \
    --duration action_partially_dependent \
    --anticipate_next --anticipate_prev --crf --crf_transition_weight 1.0 \
    --ltc_long_term_attn_g 16 --ltc_num_stages 6
```

**Train on EGTEA Gaze+ (Split 1):**
```bash
python main.py --dataset EGTEA --split 1 --seg --task multi-label --anticipate \
    --model tcca_LTC --hidden_dim 128 --hidden_dim_decoder 256 \
    --n_decoder_layer 2 --n_query 30 --max_pos_len 3100 --sample_rate 1 \
    --epochs 50 --batch_size 16 --lr 1e-3 --weight_decay 0.005 \
    --segmentation_loss cross_entropy+tmse --tsme_seg_weight 0.2 \
    --anticipate_next --anticipate_prev --anticipate_future_multilabel \
    --crf --crf_transition_weight 1.0 \
    --ltc_long_term_attn_g 16 --ltc_num_stages 4
```

**Train on EPIC-Kitchens-55 (Split 1):**
```bash
python main.py --dataset EPIC --split 1 --seg --task multi-label --anticipate \
    --model tcca_LTC --hidden_dim 64 --hidden_dim_decoder 128 \
    --n_decoder_layer 2 --n_query 250 --max_pos_len 4000 --sample_rate 2 \
    --epochs 50 --batch_size 16 --lr 1e-3 --weight_decay 0.005 \
    --segmentation_loss cross_entropy+tmse --tsme_seg_weight 0.2 \
    --anticipate_next --anticipate_prev --anticipate_future_multilabel \
    --crf --crf_transition_weight 1.0 \
    --ltc_long_term_attn_g 64 --ltc_num_stages 4
```

### Weights & Biases (W&B) Sweeps
To run hyperparameter or ablation sweeps with Weights & Biases:
1. Add your W&B API key in [`scripts/run_wandb.py`](scripts/run_wandb.py).
2. Configure the sweep parameters in [`scripts/sweeps/sweep_breakfast.yaml`](scripts/sweeps/sweep_breakfast.yaml).
3. Launch the sweep agent:
```bash
python scripts/run_wandb.py
```

---

## Repository Structure

```text
TCCA/
├── checkpoints_best/          # Pretrained TCCA model weights (Breakfast, 50Salads, EGTEA, EPIC)
├── data/                      # Dataset loader and transcript/duration preprocessing
├── docs/                      # Architecture diagrams and figures
├── model/
│   ├── tcca.py                # Main TCCA encoder-decoder architecture (BACR + duration conditioning + CRF)
│   ├── CRF/crf.py             # CRF layer (Viterbi, Stochastic FFBS, and K-Best Viterbi decoding)
│   ├── LTcontext/             # Hierarchical LTContext temporal action segmentation encoder
│   ├── FACT/                  # Frame-Action Cross-Transformer segmentation encoder
│   └── extras/                # Transformer layers, positional encodings, and TMSE smoothing loss
├── scripts/                   # Evaluation bash scripts and W&B sweep configurations
├── main.py                    # Entry point for training and evaluation
├── opts.py                    # Command-line argument definitions
├── train.py                   # Training loop (joint segmentation, BACR KL losses, duration & CRF losses)
├── predict.py                 # Inference & evaluation metrics (mMoC, Edit score, F1, Multi-label mAP)
└── utils.py                   # Evaluation and sequence processing utilities
```

---

## Citation

If you find this repository or our paper useful in your research, please cite:

```bibtex
@article{mate2024tcca,
  title={Temporal Context Consistency Above All: Enhancing Long-Term Anticipation by Learning and Enforcing Temporal Constraints},
  author={Mat{\'e}-Angulo, Alberto and Dimiccoli, Mariella},
  journal={arXiv preprint arXiv:2412.19424},
  year={2024},
  url={https://arxiv.org/abs/2412.19424}
}
```
