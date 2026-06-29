import os
import re
import sys
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_PGD_PILOTNET_DIR, RESULTS_PGD_RESNET_DIR, RESULTS_DIR, FIGURES_DIR


INPUT_FOLDERS = [str(RESULTS_PGD_PILOTNET_DIR), str(RESULTS_PGD_RESNET_DIR)]
OUTPUT_CSV = str(RESULTS_DIR / "PGD_error_distribution_long_format.csv")
OUTPUT_FOLDER = str(FIGURES_DIR / "pgd_error_distribution_plots")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def parse_filename(filename):
    name = filename.lower()

    if "pilotnet" in name:
        architecture = "PilotNet"
    elif "resnet" in name:
        architecture = "ResNet"
    else:
        architecture = "Unknown"

    if "finetuned_flipped_us_pretrained" in name or "fine_tuned_flipped" in name:
        strategy = "Fine-tuned flipped"
    elif "finetuned_us_pretrained" in name or "fine_tuned" in name:
        strategy = "Fine-tuned original"
    elif "flipped_us_trained" in name:
        strategy = "Flipped U.S. pretrained"
    elif "us_trained" in name:
        strategy = "U.S. pretrained"
    else:
        strategy = "Unknown"

    eps_match = re.search(r"eps([0-9.]+)", name)
    epsilon = float(eps_match.group(1)) if eps_match else np.nan

    return architecture, strategy, epsilon


strategy_order = [
    "U.S. pretrained",
    "Flipped U.S. pretrained",
    "Fine-tuned original",
    "Fine-tuned flipped",
]


rows = []

for folder in INPUT_FOLDERS:
    for csv_path in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
        filename = os.path.basename(csv_path)
        if "summary" in filename.lower():
            continue
        architecture, strategy, epsilon = parse_filename(filename)

        df = pd.read_csv(csv_path)

        y_true = df["Ground_Truth"].astype(float).to_numpy()
        y_clean = df["Clean_Prediction"].astype(float).to_numpy()
        y_adv = df["Adversarial_Prediction"].astype(float).to_numpy()

        clean_abs_error = np.abs(y_clean - y_true)
        adv_abs_error = np.abs(y_adv - y_true)

        for value in clean_abs_error:
            rows.append({
                "Architecture": architecture,
                "Strategy": strategy,
                "Epsilon": epsilon,
                "Condition": "Clean",
                "Absolute_Error": value
            })

        for value in adv_abs_error:
            rows.append({
                "Architecture": architecture,
                "Strategy": strategy,
                "Epsilon": epsilon,
                "Condition": "PGD",
                "Absolute_Error": value
            })


df_all = pd.DataFrame(rows)
df_all.to_csv(OUTPUT_CSV, index=False)


for architecture in sorted(df_all["Architecture"].unique()):
    for epsilon in sorted(df_all["Epsilon"].dropna().unique()):
        subset = df_all[
            (df_all["Architecture"] == architecture) &
            (np.isclose(df_all["Epsilon"], epsilon)) &
            (df_all["Condition"] == "PGD")
        ]

        if subset.empty:
            continue

        data = []
        labels = []

        for strategy in strategy_order:
            s = subset[subset["Strategy"] == strategy]["Absolute_Error"].to_numpy()
            if len(s) > 0:
                data.append(s)
                labels.append(strategy)

        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Absolute steering error under PGD (degrees)")
        plt.xlabel("Training strategy")
        plt.title(f"{architecture}, PGD ε={epsilon}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        out_path = os.path.join(
            OUTPUT_FOLDER,
            f"{architecture}_PGD_eps{epsilon}_absolute_error_boxplot.png"
        )

        plt.savefig(out_path, dpi=300)
        plt.close()

        print(f"Saved: {out_path}")
