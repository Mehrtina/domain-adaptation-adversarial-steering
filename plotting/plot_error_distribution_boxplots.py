import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_DIR, FIGURES_DIR

# -----------------------------
# Input files
# -----------------------------
fgsm_file = str(RESULTS_DIR / "FGSM_error_distribution_long_format.csv")
pgd_file  = str(RESULTS_DIR / "PGD_error_distribution_long_format.csv")
os.makedirs(FIGURES_DIR, exist_ok=True)
output_file = str(FIGURES_DIR / "Figure_error_distributions_eps003.png")

# -----------------------------
# Load data
# -----------------------------
fgsm = pd.read_csv(fgsm_file)
pgd = pd.read_csv(pgd_file)

# Keep only adversarial rows
fgsm = fgsm[fgsm["Condition"] == "FGSM"].copy()
pgd  = pgd[pgd["Condition"] == "PGD"].copy()

# Add attack labels
fgsm["Attack"] = "FGSM"
pgd["Attack"] = "PGD"

# Merge
df = pd.concat([fgsm, pgd], ignore_index=True)

# Filter epsilon = 0.03
df = df[np.isclose(df["Epsilon"], 0.03)].copy()

# Strategy order
strategy_order = [
    "U.S. pretrained",
    "Flipped U.S. pretrained",
    "Fine-tuned original",
    "Fine-tuned flipped",
]

# Shorter display labels
label_map = {
    "U.S. pretrained": "U.S. trained",
    "Flipped U.S. pretrained": "Flipped U.S. trained",
    "Fine-tuned original": "Fine-tuned U.S.-pretrained",
    "Fine-tuned flipped": "Fine-tuned flipped-U.S.-pretrained",
}

# Figure layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)

panel_specs = [
    ("PilotNet", "FGSM", axes[0, 0], "(a) PilotNet, FGSM"),
    ("PilotNet", "PGD",  axes[0, 1], "(b) PilotNet, PGD"),
    ("ResNet",   "FGSM", axes[1, 0], "(c) ResNet-18, FGSM"),
    ("ResNet",   "PGD",  axes[1, 1], "(d) ResNet-18, PGD"),
]

for architecture, attack, ax, panel_title in panel_specs:
    subset = df[
        (df["Architecture"] == architecture) &
        (df["Attack"] == attack)
    ]

    data = []
    labels = []

    for strategy in strategy_order:
        vals = subset.loc[subset["Strategy"] == strategy, "Absolute_Error"].dropna().to_numpy()
        if len(vals) > 0:
            data.append(vals)
            labels.append(label_map[strategy])

    bp = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2)
    )

    # Professional muted colors
    colors = ["#8FB9E3", "#6FA8A8", "#C8D6AF", "#8E6BBE"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylabel(f"Absolute steering error under {attack} (degrees)", fontsize=13)
    ax.set_title(panel_title, fontsize=16, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_fontsize(11)

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")

    ax.tick_params(axis="both", width=1.8, length=5)

plt.tight_layout()
plt.savefig(output_file, dpi=400, bbox_inches="tight")
plt.show()

print(f"Saved: {output_file}")