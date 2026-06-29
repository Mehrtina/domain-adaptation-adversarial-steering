import sys
import pandas as pd
import numpy as np
import glob
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_FGSM_DIR, RESULTS_DIR

input_folder = str(RESULTS_FGSM_DIR)
output_file = str(RESULTS_DIR / "FGSM_master_metrics.csv")
missing_file_report = str(RESULTS_DIR / "FGSM_missing_files.csv")

rows = []

expected = []

architectures = ["pilotnet", "resnet"]
strategies = ["us_trained", "flipped_us_trained", "finetuned_us_pretrained", "finetuned_flipped_us_pretrained"]
epsilons = ["0.01", "0.03", "0.05"]

for arch in architectures:
    for strategy in strategies:
        for eps in epsilons:
            expected.append({
                "Architecture": arch,
                "Strategy": strategy,
                "Epsilon": eps,
                "Expected_File_Pattern": f"{arch}_{strategy}_fgsm_eps{eps}.csv"
            })

files = glob.glob(os.path.join(input_folder, "**", "*.csv"), recursive=True)
files = [f for f in files if "summary" not in os.path.basename(f).lower()]
found_filenames = [os.path.basename(f).lower() for f in files]

for file_path in files:
    filename = os.path.basename(file_path).lower()

    if "pilotnet" in filename:
        architecture = "PilotNet"
    elif "resnet" in filename:
        architecture = "ResNet"
    else:
        architecture = "Unknown"

    if "finetuned_flipped_us_pretrained" in filename or "fine_tuned_flipped" in filename:
        strategy = "Fine-tuned flipped"
    elif "finetuned_us_pretrained" in filename or "fine_tuned" in filename:
        strategy = "Fine-tuned original"
    elif "flipped_us_trained" in filename:
        strategy = "Flipped U.S. pretrained"
    elif "us_trained" in filename:
        strategy = "U.S. pretrained"
    else:
        strategy = "Unknown"

    eps_match = re.search(r"eps([0-9]+(?:\.[0-9]+)?)", filename)
    epsilon = float(eps_match.group(1)) if eps_match else np.nan

    df = pd.read_csv(file_path)

    y_true = df["Ground_Truth"].to_numpy(dtype=float)
    y_clean = df["Clean_Prediction"].to_numpy(dtype=float)
    y_adv = df["Adversarial_Prediction"].to_numpy(dtype=float)

    clean_error = y_clean - y_true
    adv_error = y_adv - y_true

    clean_abs_error = np.abs(clean_error)
    adv_abs_error = np.abs(adv_error)

    clean_mse = np.mean(clean_error ** 2)
    adv_mse = np.mean(adv_error ** 2)

    clean_mae = np.mean(clean_abs_error)
    adv_mae = np.mean(adv_abs_error)

    robustness_score = clean_mse / adv_mse if adv_mse != 0 else np.nan

    exceed_5 = np.mean(adv_abs_error > 5) * 100
    exceed_10 = np.mean(adv_abs_error > 10) * 100
    exceed_15 = np.mean(adv_abs_error > 15) * 100

    rows.append({
        "Architecture": architecture,
        "Strategy": strategy,
        "Attack": "FGSM",
        "Epsilon": epsilon,
        "Clean_MSE": clean_mse,
        "Adversarial_MSE": adv_mse,
        "Clean_MAE": clean_mae,
        "Adversarial_MAE": adv_mae,
        "Robustness_Score": robustness_score,
        "Adv_Error_>5deg_%": exceed_5,
        "Adv_Error_>10deg_%": exceed_10,
        "Adv_Error_>15deg_%": exceed_15,
        "N_Test_Images": len(df),
        "Source_File": os.path.basename(file_path)
    })

master = pd.DataFrame(rows)

strategy_order = {
    "U.S. pretrained": 1,
    "Flipped U.S. pretrained": 2,
    "Fine-tuned original": 3,
    "Fine-tuned flipped": 4
}

if len(master) > 0:
    master["Strategy_Order"] = master["Strategy"].map(strategy_order)
    master = master.sort_values(
        by=["Architecture", "Epsilon", "Strategy_Order"]
    ).drop(columns=["Strategy_Order"])

os.makedirs(RESULTS_DIR, exist_ok=True)
master.to_csv(output_file, index=False)

missing_rows = []

for item in expected:
    pattern = item["Expected_File_Pattern"].lower()
    if pattern not in found_filenames:
        missing_rows.append(item)

missing_df = pd.DataFrame(missing_rows)
missing_df.to_csv(missing_file_report, index=False)

print(f"Processed files: {len(master)}")
print(f"Missing files: {len(missing_df)}")
print(f"Saved master table to: {output_file}")
print(f"Saved missing file report to: {missing_file_report}")
print(master)
