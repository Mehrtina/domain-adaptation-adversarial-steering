import os
import re
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RESULTS_FGSM_DIR, RESULTS_DIR

INPUT_FOLDER = str(RESULTS_FGSM_DIR)
OUTPUT_METRICS = str(RESULTS_DIR / "FGSM_bootstrap_CI_metrics.csv")
OUTPUT_TESTS = str(RESULTS_DIR / "FGSM_wilcoxon_significance_tests.csv")

N_BOOTSTRAP = 10000
RANDOM_SEED = 42


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

    eps_match = re.search(r"eps([0-9]+(?:\.[0-9]+)?)", filename)
    epsilon = float(eps_match.group(1)) if eps_match else np.nan

    return architecture, strategy, epsilon


def bootstrap_ci(values, func=np.mean, n_bootstrap=10000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)

    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_values.append(func(values[idx]))

    boot_values = np.asarray(boot_values)
    lower = np.percentile(boot_values, 2.5)
    upper = np.percentile(boot_values, 97.5)

    return lower, upper


def compute_metrics_for_file(csv_path):
    filename = os.path.basename(csv_path)
    architecture, strategy, epsilon = parse_filename(filename)

    df = pd.read_csv(csv_path)

    y_true = df["Ground_Truth"].astype(float).to_numpy()
    y_clean = df["Clean_Prediction"].astype(float).to_numpy()
    y_adv = df["Adversarial_Prediction"].astype(float).to_numpy()

    clean_sq_error = (y_clean - y_true) ** 2
    adv_sq_error = (y_adv - y_true) ** 2

    clean_abs_error = np.abs(y_clean - y_true)
    adv_abs_error = np.abs(y_adv - y_true)

    clean_mse = np.mean(clean_sq_error)
    adv_mse = np.mean(adv_sq_error)
    clean_mae = np.mean(clean_abs_error)
    adv_mae = np.mean(adv_abs_error)
    rs = clean_mse / adv_mse if adv_mse != 0 else np.nan

    clean_mse_ci = bootstrap_ci(clean_sq_error, np.mean, N_BOOTSTRAP, RANDOM_SEED)
    adv_mse_ci = bootstrap_ci(adv_sq_error, np.mean, N_BOOTSTRAP, RANDOM_SEED)
    clean_mae_ci = bootstrap_ci(clean_abs_error, np.mean, N_BOOTSTRAP, RANDOM_SEED)
    adv_mae_ci = bootstrap_ci(adv_abs_error, np.mean, N_BOOTSTRAP, RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    rs_boot = []

    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_clean_mse = np.mean(clean_sq_error[idx])
        boot_adv_mse = np.mean(adv_sq_error[idx])
        rs_boot.append(boot_clean_mse / boot_adv_mse if boot_adv_mse != 0 else np.nan)

    rs_boot = np.asarray(rs_boot)
    rs_ci_lower = np.nanpercentile(rs_boot, 2.5)
    rs_ci_upper = np.nanpercentile(rs_boot, 97.5)

    return {
        "Architecture": architecture,
        "Strategy": strategy,
        "Attack": "FGSM",
        "Epsilon": epsilon,
        "N": len(df),

        "Clean_MSE": clean_mse,
        "Clean_MSE_CI95_Lower": clean_mse_ci[0],
        "Clean_MSE_CI95_Upper": clean_mse_ci[1],

        "Adversarial_MSE": adv_mse,
        "Adversarial_MSE_CI95_Lower": adv_mse_ci[0],
        "Adversarial_MSE_CI95_Upper": adv_mse_ci[1],

        "Clean_MAE": clean_mae,
        "Clean_MAE_CI95_Lower": clean_mae_ci[0],
        "Clean_MAE_CI95_Upper": clean_mae_ci[1],

        "Adversarial_MAE": adv_mae,
        "Adversarial_MAE_CI95_Lower": adv_mae_ci[0],
        "Adversarial_MAE_CI95_Upper": adv_mae_ci[1],

        "Robustness_Score": rs,
        "RS_CI95_Lower": rs_ci_lower,
        "RS_CI95_Upper": rs_ci_upper,

        "Adv_Error_>5deg_%": np.mean(adv_abs_error > 5) * 100,
        "Adv_Error_>10deg_%": np.mean(adv_abs_error > 10) * 100,
        "Adv_Error_>15deg_%": np.mean(adv_abs_error > 15) * 100,

        "Source_File": filename,
    }


def load_errors(csv_path):
    df = pd.read_csv(csv_path)

    y_true = df["Ground_Truth"].astype(float).to_numpy()
    y_adv = df["Adversarial_Prediction"].astype(float).to_numpy()

    frames = df["Frame"].to_numpy()
    adv_abs_error = np.abs(y_adv - y_true)

    return pd.DataFrame({
        "Frame": frames,
        "Adv_Abs_Error": adv_abs_error
    })


def run_significance_tests(csv_files):
    records = []

    key_epsilon = 0.03

    comparisons = [
        ("U.S. pretrained", "Fine-tuned original"),
        ("Flipped U.S. pretrained", "Fine-tuned flipped"),
        ("Fine-tuned original", "Fine-tuned flipped"),
    ]

    file_index = {}

    for path in csv_files:
        filename = os.path.basename(path)
        architecture, strategy, epsilon = parse_filename(filename)

        if np.isclose(epsilon, key_epsilon):
            file_index[(architecture, strategy)] = path

    for architecture in ["PilotNet", "ResNet"]:
        for strategy_a, strategy_b in comparisons:
            key_a = (architecture, strategy_a)
            key_b = (architecture, strategy_b)

            if key_a not in file_index or key_b not in file_index:
                records.append({
                    "Architecture": architecture,
                    "Epsilon": key_epsilon,
                    "Comparison": f"{strategy_a} vs {strategy_b}",
                    "Test": "Wilcoxon signed-rank",
                    "Statistic": np.nan,
                    "p_value": np.nan,
                    "N_Paired_Frames": 0,
                    "Note": "Missing one or both files"
                })
                continue

            errors_a = load_errors(file_index[key_a])
            errors_b = load_errors(file_index[key_b])

            merged = errors_a.merge(
                errors_b,
                on="Frame",
                suffixes=("_A", "_B")
            )

            if len(merged) == 0:
                records.append({
                    "Architecture": architecture,
                    "Epsilon": key_epsilon,
                    "Comparison": f"{strategy_a} vs {strategy_b}",
                    "Test": "Wilcoxon signed-rank",
                    "Statistic": np.nan,
                    "p_value": np.nan,
                    "N_Paired_Frames": 0,
                    "Note": "No paired frames"
                })
                continue

            diff = merged["Adv_Abs_Error_A"] - merged["Adv_Abs_Error_B"]

            if np.allclose(diff, 0):
                statistic = np.nan
                p_value = 1.0
                note = "All paired differences are zero"
            else:
                statistic, p_value = wilcoxon(
                    merged["Adv_Abs_Error_A"],
                    merged["Adv_Abs_Error_B"],
                    zero_method="wilcox",
                    alternative="two-sided"
                )
                note = ""

            records.append({
                "Architecture": architecture,
                "Epsilon": key_epsilon,
                "Comparison": f"{strategy_a} vs {strategy_b}",
                "Test": "Wilcoxon signed-rank",
                "Statistic": statistic,
                "p_value": p_value,
                "N_Paired_Frames": len(merged),
                "Mean_AdvAbsError_A": merged["Adv_Abs_Error_A"].mean(),
                "Mean_AdvAbsError_B": merged["Adv_Abs_Error_B"].mean(),
                "Median_AdvAbsError_A": merged["Adv_Abs_Error_A"].median(),
                "Median_AdvAbsError_B": merged["Adv_Abs_Error_B"].median(),
                "Note": note
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    all_files = glob.glob(os.path.join(INPUT_FOLDER, "**", "*.csv"), recursive=True)
    csv_files = sorted(f for f in all_files if "summary" not in os.path.basename(f).lower())

    print(f"Found {len(csv_files)} CSV files")

    metric_rows = []

    for csv_path in csv_files:
        print(f"Processing {os.path.basename(csv_path)}")
        metric_rows.append(compute_metrics_for_file(csv_path))

    metrics_df = pd.DataFrame(metric_rows)

    strategy_order = {
        "U.S. pretrained": 1,
        "Flipped U.S. pretrained": 2,
        "Fine-tuned original": 3,
        "Fine-tuned flipped": 4,
    }

    metrics_df["Strategy_Order"] = metrics_df["Strategy"].map(strategy_order)
    metrics_df = metrics_df.sort_values(
        by=["Architecture", "Epsilon", "Strategy_Order"]
    ).drop(columns=["Strategy_Order"])

    metrics_df.to_csv(OUTPUT_METRICS, index=False)
    print(f"Saved bootstrap metrics to: {OUTPUT_METRICS}")

    tests_df = run_significance_tests(csv_files)
    tests_df.to_csv(OUTPUT_TESTS, index=False)
    print(f"Saved significance tests to: {OUTPUT_TESTS}")