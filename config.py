"""
Central configuration for the adversarial robustness study. "Domain Adaptation Affects Adversarial Robustness in Autonomous Steering Models"

All data paths, weight paths, and output directories are defined here.
Scripts import these constants rather than hard-coding paths.

Quick-start layout expected under the project root:
  data_files/
    test_images/        ← Australian test-set JPEG frames
    test_labels.csv     ← ground-truth steering angles (degrees)
    us_training/        ← U.S. training images + data.txt
    finetuning/         ← Fine-tuning images + totalDATA.txt
  weights/              ← .weights.h5 model files (see weights/README.md)

Edit the WEIGHTS_* paths below if your weight files are stored elsewhere.
"""

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent

# ── Test data (Australian held-out set) ──────────────────────────────────────
TEST_IMAGE_DIR = PROJECT_ROOT / "data_files" / "test_images"
TEST_LABEL_CSV = PROJECT_ROOT / "data_files" / "test_labels.csv"

# ── Fine-tuning data ──────────────────────────────────────────────────────────
FINETUNE_DATA_DIR = PROJECT_ROOT / "data_files" / "finetuning"

# ── U.S. training data ────────────────────────────────────────────────────────
US_TRAINING_DIR = PROJECT_ROOT / "data_files" / "us_training"
DATA_DIR = US_TRAINING_DIR  

# ── Weight files ──────────────────────────────────────────────────────────────
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# U.S.-trained weights
WEIGHTS_PILOTNET_US       = WEIGHTS_DIR / "pretrained_pilotnet.weights.h5"
WEIGHTS_PILOTNET_FLIPPED  = WEIGHTS_DIR / "flipped_pretrained_pilotnet.weights.h5"
WEIGHTS_RESNET_US         = WEIGHTS_DIR / "pretrained_resnet.weights.h5"
WEIGHTS_RESNET_FLIPPED    = WEIGHTS_DIR / "flipped_pretrained_resnet.weights.h5"

# Fine-tuned (partial-backbone) weights -> main manuscript
WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED = WEIGHTS_DIR / "pilotnet_finetuned_flipped.weights.h5"
WEIGHTS_PILOTNET_PARTIAL_FT_US      = WEIGHTS_DIR / "pilotnet_finetuned_us.weights.h5"
WEIGHTS_RESNET_PARTIAL_FT_US        = WEIGHTS_DIR / "resnet_finetuned_us.weights.h5"
WEIGHTS_RESNET_PARTIAL_FT_FLIPPED   = WEIGHTS_DIR / "resnet_finetuned_flipped.weights.h5"


# Fine-tuned (frozen-backbone / head-only) weights -> Supplementary
WEIGHTS_PILOTNET_FT_HEAD_ONLY_US      = WEIGHTS_DIR / "pilotnet_frozen_backbone_finetuned.weights.h5"
WEIGHTS_PILOTNET_FT_HEAD_ONLY_FLIPPED = WEIGHTS_DIR / "pilotnet_frozen_backbone_finetuned_flipped.weights.h5"
WEIGHTS_RESNET_FT_HEAD_ONLY_US        = WEIGHTS_DIR / "resnet_frozen_backbone_finetuned.weights.h5"
WEIGHTS_RESNET_FT_HEAD_ONLY_FLIPPED   = WEIGHTS_DIR / "resnet_frozen_backbone_finetuned_flipped.weights.h5"


# ── Output directories ────────────────────────────────────────────────────────
LOGS_DIR    = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Attack result subdirectories (written by attacks/*.py, read by analysis/*.py)
RESULTS_FGSM_DIR         = RESULTS_DIR / "fgsm_evaluation"
RESULTS_PGD_PILOTNET_DIR = RESULTS_DIR / "pgd_pilotnet"
RESULTS_PGD_RESNET_DIR   = RESULTS_DIR / "pgd_resnet"

# ── Training hyperparameters (shared defaults) ────────────────────────────────
BATCH_SIZE    = 128
NUM_EPOCHS    = 50
NUM_EPOCHS_FT = 15
LEARNING_RATE = 1e-4

# ── Additional train_pilotnet.py settings ────────────────────────────────────
LOG_DIR_TRAIN_PILOTNET         = LOGS_DIR / "pilotnet_logs"
LOG_DIR_TRAIN_RESNET         = LOGS_DIR / "resnet_logs"

L2_REG_CONST          = 1e-3
