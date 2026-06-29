# Domain Adaptation Affects Adversarial Robustness in Autonomous Steering Models
Associated with a manuscript submitted to *Scientific Reports* — Domain Adaptation Affects Adversarial Robustness in Autonomous Steering Models

This repository contains the code used to generate the analyses, figures, and supplementary results described in the manuscript.

---

## Overview

We evaluate the adversarial robustness of two end-to-end deep-learning steering-angle predictors — **PilotNet** and **ResNet-18** — when tested on Australian highway footage after training on U.S. data (domain shift). Four training strategies are compared under **FGSM** and **PGD** attacks at three perturbation levels. All perturbations are applied in normalised YUV input space ([0, 1]).

**Key experimental dimensions:**

| Dimension | Values |
|-----------|--------|
| Architectures | PilotNet, ResNet-18 |
| Training strategies | U.S.-trained, Flipped-U.S.-trained, Fine-tuned U.S.-pretrained, Fine-tuned Flipped-U.S.-pretrained |
| Attacks | FGSM, PGD |
| Epsilon (ε) | 0.01, 0.03, 0.05 |
| PGD alpha (α) | 0.002, 0.005, 0.010 (paired with ε) |
| PGD iterations | 10 |
| Test set | 3,000 Australian highway frames |

**Steering unit flow:**
Ground-truth labels are stored in **degrees** → converted to **radians** for model training loss and adversarial attack loss → model predictions output in **radians** → converted back to **degrees** before writing per-frame CSVs → all reported metrics (MSE, MAE, threshold exceedance) are in **degrees**.

**Fine-tuning strategies:**
This repository includes both partial-backbone fine-tuning and frozen-backbone/head-only fine-tuning scripts. The partial-backbone fine-tuning scripts correspond to earlier adaptation experiments, while the frozen-backbone/head-only scripts were added during revision to provide a more controlled adaptation setting. In the frozen-backbone setting, the convolutional feature extractor is kept fixed and only the final regression head is trained on the Australian fine-tuning data. Scripts are named to indicate whether they use partial-backbone or frozen-backbone fine-tuning.

---

## Repository Structure

```
.
├── config.py                              # Central path and hyperparameter configuration
├── preprocessing/
│   ├── preprocess_us_training.py          # 150-px crop; used for U.S. training data
│   └── preprocess_australian.py          # 800-px crop; used for all evaluation and fine-tuning
├── models/
│   ├── pilotnet.py                        # PilotNet
│   └── resnet18.py                        # ResNet-18 
├── data/
│   ├── dataloader.py                      # U.S. training data loader
│   └── dataloader_finetune.py             # Australian fine-tuning data loader
├── training/
│   ├── train_pilotnet.py                              # Train PilotNet from scratch on U.S. data
│   ├── train_resnet.py                                # Train ResNet-18 from scratch on U.S. data
│   ├── finetune_pilotnet_frozen_backbone.py        # Fine-tune PilotNet → AUS (frozen backbone)
│   ├── finetune_pilotnet_partial_backbone.py  # Exploratory partial-freeze variant (PilotNet)
│   ├── finetune_resnet_frozen_backbone.py          # Fine-tune ResNet-18 → AUS (frozen backbone)
│   └── finetune_resnet_partial_backbone.py         # Exploratory partial-freeze variant (ResNet-18)
├── evaluation/                               # Manuscript adversarial evaluation scripts
│   ├── evaluate_fgsm.py                   # FGSM — all strategies × both architectures
│   ├── evaluate_pgd_pilotnet.py           # PGD — PilotNet, all strategies
│   └── evaluate_pgd_resnet.py             # PGD — ResNet-18, all strategies
│   ├── evaluate.py                        # evaluates clean, non-adversarial steering predictions on the test set
├── analysis/
│   ├── compute_fgsm_metrics.py            # Aggregate FGSM per-frame CSVs into master table
│   ├── compute_pgd_metrics.py             # Aggregate PGD per-frame CSVs into master table
│   ├── bootstrap_significance_fgsm.py     # Bootstrap 95% CI + Wilcoxon test, FGSM
│   ├── bootstrap_significance_pgd.py      # Bootstrap 95% CI + Wilcoxon test, PGD
│   ├── pgd_step_size_sensitivity.py               # Supplementary: PGD step-size sensitivity study
│   └── input_gradient_sensitivity.py              # Supplementary: input-gradient norm analysis
├── plotting/
│   ├── build_fgsm_error_distribution.py  # Build FGSM long-format CSV for figures
│   ├── build_pgd_error_distribution.py   # Build PGD long-format CSV for figures
│   ├── plot_error_distribution_boxplots.py # 2×2 error-distribution boxen-plot panel
│   ├── plot_ecdf.py                       # 2×2 ECDF panel (ε = 0.03)
│   └── plot_steering_angle_histogram.py   # Test-set steering-angle distribution
├── weights/README.md                      # Where to place model weight files
├── data_files                             # Dataset path
├── results/README.md                      # Output directory structure
├── requirements.txt
└── LICENSE
```

---

## Installation

Python 3.9 or 3.10 is recommended.

```bash
git clone https://github.com/Mehrtina/domain-adaptation-adversarial-steering.git
cd domain-adaptation-adversarial-steering
pip install -r requirements.txt
```

**Run all scripts from the project root directory.**
Each script adds the project root to `sys.path` automatically.

---

## Configuration

All paths are centralised in `config.py`. The expected layout under the project root is:

```
data_files/
  test_images/        ← Australian test-set JPEG frames
  test_labels.csv     ← ground-truth steering angles (degrees)
  us_training/        ← U.S. training images + data.txt
  finetuning/         ← Fine-tuning images + totalDATA.txt
weights/              ← .weights.h5 model files (see weights/README.md)
```

If your local directories have different names, update the corresponding constants in `config.py` rather than editing individual scripts.

---

## Data and Model Weights

> Scripts will not run without the relevant data and weight files in place.

### Test set
The Australian highway test set (3,000 frames) and ground-truth CSV are needed to run `evaluation/`, `analysis/`, and `plotting/` scripts.

**The test set is not redistributed with this repository because it is subject to institutional data-use restrictions at our research centre.**

### Training / fine-tuning data
Datasets are not distributed with this repository. Refer to the manuscript for U.S. dataset (NVIDIA) citations and sources. Update `config.py` with local paths.

### Model weights
Pre-trained and fine-tuned weight files (`.weights.h5`) are large and provided separately.

See `weights/README.md` for the full list of expected files.

---

## Reproducing the Manuscript Results
The manuscript results can be reproduced from the trained weights and test data. Training and fine-tuning scripts are included for completeness but are optional if the provided weights are used.

### 1 — Train from scratch or Prepare data and weights

For training from scratch:
```bash
python training/train_pilotnet.py 
python training/train_resnet.py 
```
And for finetuning:
```bash
python training/finetune_pilotnet_partial_backbone.py
python training/finetune_resnet_partial_backbone.py
```

Place the test data and model weights in the paths defined in config.py, or edit config.py to point to your local files.

data_files/test_images/

data_files/test_labels.csv

weights/

### 2 — Run adversarial evaluations

```bash
python evaluation/evaluate_fgsm.py 
python evaluation/evaluate_pgd_pilotnet.py
python evaluation/evaluate_pgd_resnet.py
```
FGSM is evaluated at ε = 0.01, 0.03, and 0.05. PGD uses ε/α pairs of 0.01/0.002, 0.03/0.005, and 0.05/0.010, with 10 iterations.


### 3 — Compute metrics and statistical tests


```bash
python analysis/compute_fgsm_metrics.py
python analysis/compute_pgd_metrics.py 
python analysis/bootstrap_significance_fgsm.py
python analysis/bootstrap_significance_pgd.py
```
These scripts compute the main metric tables, bootstrap confidence intervals, and Wilcoxon signed-rank tests.


### 4 — Run supplementary analyses

```bash
python analysis/pgd_step_size_sensitivity.py 
python analysis/input_gradient_sensitivity.py
```

### 5 — Generate figures

```bash
python plotting/build_fgsm_error_distribution.py 
python plotting/build_pgd_error_distribution.py
python plotting/plot_error_distribution_boxplots.py 
python plotting/plot_ecdf.py 
python plotting/plot_steering_angle_histogram.py
```

---

## Metrics

All error metrics are computed in **degrees**:

| Metric | Definition |
|--------|-----------|
| **MSE** | Mean squared error between adversarial predictions and ground truth |
| **MAE** | Mean absolute error between adversarial predictions and ground truth |
| **RMSE** | √MSE |
| **Robustness score** | MSE(clean) / MSE(adversarial) — values closer to 0 indicate greater degradation under attack; values closer to 1 indicate the attack caused minimal additional error relative to the clean baseline |
| **Threshold exceedance** | % of test images where \|adversarial error\| > 5°, 10°, 15° |
| **Bootstrap 95% CI** | Percentile bootstrap, 10,000 resamplings, seed 42 |
| **Wilcoxon signed-rank** | Two-sided paired test on per-frame adversarial absolute errors at ε = 0.03 |

---

## Reproducing the Manuscript Results

- Random seed: `RANDOM_SEED = 42` (set for NumPy and TensorFlow in all attack scripts)
- PGD initialisation: uniform random perturbation in [−ε/2, +ε/2] before the first gradient step
- All perturbations are clipped to [0, 1] at every PGD step and after FGSM
- **Unit flow summary:** degrees (CSV) → radians (training loss, attack loss) → radians (model output) → degrees (per-frame CSV, metrics)
- `preprocessing/preprocess_australian.py` crops the bottom 800 pixels before resizing; this is the active preprocessing for all evaluation and fine-tuning scripts
- `preprocessing/preprocess_us_training.py` crops 150 pixels; used for U.S. training data loaders only

---

## Citation

The citation for this repository will be added after the associated manuscript has been accepted and published.

---

## License

This code is released under the MIT License. See `LICENSE` for details.
---

## Contact

For questions about the code or data, please open a GitHub issue or contact the corresponding author at z.mehraban@qut.edu.au.
