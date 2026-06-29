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
pgd_file = str(RESULTS_DIR / "PGD_error_distribution_long_format.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)
output_file = str(FIGURES_DIR / "Figure_ECDF_absolute_error_eps0.03.png")

# -----------------------------
# Load data
# -----------------------------
fgsm = pd.read_csv(fgsm_file)
pgd = pd.read_csv(pgd_file)

# Keep adversarial rows only
fgsm = fgsm[fgsm["Condition"] == "FGSM"].copy()
pgd = pgd[pgd["Condition"] == "PGD"].copy()

# Add attack labels
fgsm["Attack"] = "FGSM"
pgd["Attack"] = "PGD"

# Combine
df = pd.concat([fgsm, pgd], ignore_index=True)

# Filter epsilon = 0.03
df = df[np.isclose(df["Epsilon"], 0.03)].copy()

# Standardize architecture names if needed
df["Architecture"] = df["Architecture"].replace({
    "ResNet": "ResNet-18"
})

strategy_order = [
    "U.S. pretrained",
    "Flipped U.S. pretrained",
    "Fine-tuned original",
    "Fine-tuned flipped",
]

strategy_labels = {
    "U.S. pretrained": "U.S. trained",
    "Flipped U.S. pretrained": "Flipped U.S. trained",
    "Fine-tuned original": "Fine-tuned U.S.-pretrained",
    "Fine-tuned flipped": "Fine-tuned flipped-U.S.-pretrained",
}

colors = {
    "U.S. pretrained": "#4C78A8",
    "Flipped U.S. pretrained": "#F58518",
    "Fine-tuned original": "#54A24B",
    "Fine-tuned flipped": "#B279A2",
}

def plot_ecdf(ax, subset, title):
    for strategy in strategy_order:
        values = subset.loc[
            subset["Strategy"] == strategy, "Absolute_Error"
        ].dropna().to_numpy()

        if len(values) == 0:
            continue

        values = np.sort(values)
        y = np.arange(1, len(values) + 1) / len(values)

        ax.step(
            values,
            y,
            where="post",
            linewidth=3,
            label=strategy_labels[strategy],
            color=colors[strategy]
        )

    # Safety-relevant reference thresholds
    for threshold in [5, 10, 15]:
        ax.axvline(threshold, linestyle="--", linewidth=1, alpha=0.5)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("Absolute adversarial steering error (degrees)", fontsize=15)
    ax.set_ylabel("Fraction of test images ≤ error", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylim(0, 1.01)

# -----------------------------
# Make 2x2 ECDF figure
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)

panels = [
    ("PilotNet", "FGSM", axes[0, 0], "(a) PilotNet, FGSM"),
    ("PilotNet", "PGD", axes[0, 1], "(b) PilotNet, PGD"),
    ("ResNet-18", "FGSM", axes[1, 0], "(c) ResNet-18, FGSM"),
    ("ResNet-18", "PGD", axes[1, 1], "(d) ResNet-18, PGD"),
]

for architecture, attack, ax, title in panels:
    subset = df[
        (df["Architecture"] == architecture) &
        (df["Attack"] == attack)
    ]
    plot_ecdf(ax, subset, title)

# One shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=True,
    bbox_to_anchor=(0.5, 0.99),
    prop={"weight": "bold", "size": 13},
    edgecolor="black",
)

plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(output_file, dpi=400, bbox_inches="tight")
plt.show()

print(f"Saved: {output_file}")