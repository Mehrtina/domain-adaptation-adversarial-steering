"""
Input-gradient sensitivity analysis for PilotNet and ResNet-18.

Computes gradient of the clean MSE loss with respect to the input image
for each test frame, then reports L1 / L2 / L-inf / mean-absolute norms
per model strategy. Summarised in a master CSV alongside per-frame CSVs.
"""
import os
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from preprocessing.preprocess_australian import preprocess_image
from models.pilotnet import PilotNet
from models.resnet18 import ResNet18Steering
from config import (
    TEST_IMAGE_DIR, TEST_LABEL_CSV, RESULTS_DIR,
    WEIGHTS_PILOTNET_US, WEIGHTS_PILOTNET_FLIPPED,
    WEIGHTS_PILOTNET_PARTIAL_FT_US, WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED,
    WEIGHTS_RESNET_US, WEIGHTS_RESNET_FLIPPED,
    WEIGHTS_RESNET_PARTIAL_FT_US, WEIGHTS_RESNET_PARTIAL_FT_FLIPPED,
)



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_ground_truth(csv_path):
    ground_truth = {}

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)

        print("CSV Headers:", headers)

        if len(headers) == 1:
            for row in reader:
                if not row:
                    continue

                parts = row[0].split()

                if len(parts) >= 2:
                    image_name = parts[0]
                    steering_angle = parts[1]

                    try:
                        image_index = int(
                            image_name
                            .replace("frame_", "")
                            .replace(".jpg", "")
                            .split(".")[0]
                        )
                        ground_truth[image_index] = float(steering_angle)

                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")

        else:
            for row in reader:
                if len(row) < 2:
                    print(f"Skipping row due to insufficient data: {row}")
                    continue

                try:
                    image_index = int(
                        str(row[0])
                        .replace("frame_", "")
                        .replace(".jpg", "")
                        .split(".")[0]
                    )
                    ground_truth[image_index] = float(row[1])

                except ValueError as e:
                    print(f"Skipping row due to error: {e}")

    return ground_truth


def build_model(architecture):
    if architecture == "PilotNet":
        return PilotNet(input_shape=(66, 200, 3)).build_model()

    if architecture == "ResNet-18":
        return ResNet18Steering(input_shape=(66, 200, 3)).build_model()

    raise ValueError(f"Unknown architecture: {architecture}")


def bootstrap_ci(values, n_bootstrap=1000, ci=95, seed=42):
    """Return bootstrap CI for the mean."""
    values = np.array(values, dtype=np.float64)

    if len(values) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))

    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)

    return lower, upper


def compute_input_gradient_norms(model, image, label_degrees):


    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

  
    image_tensor = tf.expand_dims(image_tensor, axis=0)

    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor, training=False)
        loss = tf.reduce_mean(tf.square(prediction - label_tensor))

    gradients = tape.gradient(loss, image_tensor)

    if gradients is None:
        return np.nan, np.nan, np.nan, np.nan

    grad_np = gradients.numpy().astype(np.float64)

    flat_grad = grad_np.reshape(-1)

    l1_norm = np.sum(np.abs(flat_grad))
    l2_norm = np.sqrt(np.sum(flat_grad ** 2))
    linf_norm = np.max(np.abs(flat_grad))
    mean_abs_grad = np.mean(np.abs(flat_grad))

    return l1_norm, l2_norm, linf_norm, mean_abs_grad


def evaluate_input_gradient_sensitivity(
    architecture,
    data_dir,
    weights_path,
    save_folder,
    ground_truth,
    strategy_short,
):
    os.makedirs(save_folder, exist_ok=True)

    model = build_model(architecture)
    model.load_weights(weights_path)

    arch_file_prefix = "pilotnet" if architecture == "PilotNet" else "resnet"

    per_image_csv_path = os.path.join(
        save_folder,
        f"{arch_file_prefix}_{strategy_short}_input_gradient_norms.csv"
    )

    l1_values = []
    l2_values = []
    linf_values = []
    mean_abs_values = []

    with open(per_image_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Frame",
                "Ground_Truth",
                "InputGrad_L1",
                "InputGrad_L2",
                "InputGrad_Linf",
                "InputGrad_MeanAbs",
                "Valid_Gradient",
            ],
        )
        writer.writeheader()

        for frame_number in ground_truth.keys():
            image_name = f"frame_{frame_number}.jpg"
            image_path = os.path.join(data_dir, image_name)

            if not os.path.exists(image_path):
                continue

            actual_steering = ground_truth.get(frame_number)

            if actual_steering is None:
                continue

            processed_image = preprocess_image(image_path)

            l1_norm, l2_norm, linf_norm, mean_abs_grad = compute_input_gradient_norms(
                model=model,
                image=processed_image,
                label_degrees=actual_steering,
            )
            valid_gradient = not any(
                np.isnan(v) for v in [l1_norm, l2_norm, linf_norm, mean_abs_grad]
            )

            writer.writerow({
                "Frame": frame_number,
                "Ground_Truth": actual_steering,
                "InputGrad_L1": l1_norm,
                "InputGrad_L2": l2_norm,
                "InputGrad_Linf": linf_norm,
                "InputGrad_MeanAbs": mean_abs_grad,
                "Valid_Gradient": valid_gradient,
            })

            if valid_gradient:
                l1_values.append(l1_norm)
                l2_values.append(l2_norm)
                linf_values.append(linf_norm)
                mean_abs_values.append(mean_abs_grad)

    l1_values = np.array(l1_values)
    l2_values = np.array(l2_values)
    linf_values = np.array(linf_values)
    mean_abs_values = np.array(mean_abs_values)

    l2_ci_low, l2_ci_high = bootstrap_ci(l2_values)

    metrics = {
        "N": len(l2_values),

        "Mean_L1": np.mean(l1_values),
        "Median_L1": np.median(l1_values),

        "Mean_L2": np.mean(l2_values),
        "Median_L2": np.median(l2_values),
        "L2_95CI_Lower": l2_ci_low,
        "L2_95CI_Upper": l2_ci_high,

        "Mean_Linf": np.mean(linf_values),
        "Median_Linf": np.median(linf_values),

        "Mean_Abs_Gradient": np.mean(mean_abs_values),
        "Median_Abs_Gradient": np.median(mean_abs_values),
    }

    summary_txt_path = os.path.join(
        save_folder,
        f"{arch_file_prefix}_{strategy_short}_input_gradient_summary.txt"
    )

    with open(summary_txt_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Completed {architecture} {strategy_short}")
    print(metrics)

    tf.keras.backend.clear_session()

    return metrics


if __name__ == "__main__":

    DATA_DIR = str(TEST_IMAGE_DIR)
    GROUND_TRUTH_PATH = str(TEST_LABEL_CSV)

    OUTPUT_ROOT = str(RESULTS_DIR / "input_gradient_sensitivity")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    experiments = [

        {
            "architecture": "PilotNet",
            "strategy": "U.S. pretrained",
            "short_name": "us_trained",
            "weights": str(WEIGHTS_PILOTNET_US),
        },
        {
            "architecture": "PilotNet",
            "strategy": "Flipped U.S. pretrained",
            "short_name": "flipped_us_trained",
            "weights": str(WEIGHTS_PILOTNET_FLIPPED),
        },
        {
            "architecture": "PilotNet",
            "strategy": "Fine-tuned from U.S.-pretrained",
            "short_name": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_PARTIAL_FT_US),
        },
        {
            "architecture": "PilotNet",
            "strategy": "Fine-tuned from Flipped-U.S.-pretrained",
            "short_name": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED),
        },

     
        {
            "architecture": "ResNet-18",
            "strategy": "U.S. pretrained",
            "short_name": "us_trained",
            "weights": str(WEIGHTS_RESNET_US),
        },
        {
            "architecture": "ResNet-18",
            "strategy": "Flipped U.S. pretrained",
            "short_name": "flipped_us_trained",
            "weights": str(WEIGHTS_RESNET_FLIPPED),
        },
        {
            "architecture": "ResNet-18",
            "strategy": "Fine-tuned from U.S.-pretrained",
            "short_name": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_RESNET_PARTIAL_FT_US),
        },
        {
            "architecture": "ResNet-18",
            "strategy": "Fine-tuned from Flipped-U.S.-pretrained",
            "short_name": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_RESNET_PARTIAL_FT_FLIPPED),
        },
    ]

    summary_rows = []

    for exp in experiments:
        if not os.path.exists(exp["weights"]):
            print(f"WARNING: Missing weights file: {exp['weights']}")
            continue

        print("=" * 90)
        print(
            f"Running input-gradient sensitivity | "
            f"{exp['architecture']} | {exp['strategy']}"
        )
        print("Weights:", exp["weights"])

        arch_folder = "PilotNet" if exp["architecture"] == "PilotNet" else "ResNet18"

        save_folder = os.path.join(
            OUTPUT_ROOT,
            arch_folder,
            exp["short_name"]
        )

        metrics = evaluate_input_gradient_sensitivity(
            architecture=exp["architecture"],
            data_dir=DATA_DIR,
            weights_path=exp["weights"],
            save_folder=save_folder,
            ground_truth=ground_truth,
            strategy_short=exp["short_name"],
        )

        row = {
            "Architecture": exp["architecture"],
            "Strategy": exp["strategy"],
            **metrics,
        }

        summary_rows.append(row)

    summary_csv_path = os.path.join(
        OUTPUT_ROOT,
        "input_gradient_sensitivity_summary.csv"
    )

    fieldnames = [
        "Architecture",
        "Strategy",
        "N",
        "Mean_L1",
        "Median_L1",
        "Mean_L2",
        "Median_L2",
        "L2_95CI_Lower",
        "L2_95CI_Upper",
        "Mean_Linf",
        "Median_Linf",
        "Mean_Abs_Gradient",
        "Median_Abs_Gradient",
    ]

    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 90)
    print("Input-gradient sensitivity analysis complete.")
    print(f"Summary saved to: {summary_csv_path}")
