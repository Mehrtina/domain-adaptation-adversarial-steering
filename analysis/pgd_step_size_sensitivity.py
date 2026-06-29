"""
PGD step-size sensitivity analysis for PilotNet and ResNet-18.

Keeps epsilon fixed at 0.03 and varies only the step size (alpha) across
{0.002, 0.005, 0.010} for all 8 model variants. Outputs per-frame CSVs and
a master summary CSV.
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
    WEIGHTS_PILOTNET_FT_US, WEIGHTS_PILOTNET_FT_FLIPPED,
    WEIGHTS_RESNET_US, WEIGHTS_RESNET_FLIPPED,
    WEIGHTS_RESNET_FT_US, WEIGHTS_RESNET_FT_FLIPPED,
)


# ============================================================
# Settings
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

ITERATIONS = 10

# PGD step-size sensitivity:
# keep epsilon fixed and vary only alpha
SENSITIVITY_EPSILON = 0.03
SENSITIVITY_ALPHAS = [0.002, 0.005, 0.010]


# ============================================================
# PGD attack: unit-corrected
# ============================================================

def pgd_attack(model, image, label_degrees, epsilon=0.03, alpha=0.005, iterations=10):
    """
    PGD attack with unit-corrected loss.

    Ground-truth labels in CSV: degrees
    Model output: radians
    Attack loss: radians
    Reported predictions: degrees
    """

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    # Convert label from degrees to radians for attack loss
    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

    # Random initialization inside epsilon-ball
    adv_image = image_tensor + tf.random.uniform(
        image_tensor.shape,
        minval=-epsilon / 2,
        maxval=epsilon / 2,
        dtype=tf.float32,
    )
    adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    for _ in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(tf.expand_dims(adv_image, axis=0), training=False)
            loss = tf.keras.losses.MeanSquaredError()(label_tensor, prediction)

        gradients = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradients)

        adv_image = adv_image + alpha * signed_grad

        # Project perturbation back into L-infinity epsilon-ball
        perturbation = tf.clip_by_value(
            adv_image - image_tensor,
            -epsilon,
            epsilon
        )
        adv_image = image_tensor + perturbation

        # Keep valid image range
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    return adv_image


# ============================================================
# Load ground truth
# ============================================================

def load_ground_truth(csv_path):
    """Load ground-truth steering angles from CSV."""
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


# ============================================================
# Model builder
# ============================================================

def build_model(architecture):
    if architecture == "PilotNet":
        return PilotNet(input_shape=(66, 200, 3)).build_model()

    if architecture == "ResNet-18":
        return ResNet18Steering(input_shape=(66, 200, 3)).build_model()

    raise ValueError(f"Unknown architecture: {architecture}")


# ============================================================
# Metric calculation
# ============================================================

def compute_metrics(predictions, adversarial_predictions, actuals):
    predictions = np.array(predictions)
    adversarial_predictions = np.array(adversarial_predictions)
    actuals = np.array(actuals)

    clean_errors = predictions - actuals
    adversarial_errors = adversarial_predictions - actuals
    abs_adv_errors = np.abs(adversarial_errors)

    mse_clean = np.mean(clean_errors ** 2)
    mae_clean = np.mean(np.abs(clean_errors))

    mse_adv = np.mean(adversarial_errors ** 2)
    mae_adv = np.mean(abs_adv_errors)

    robustness_score = mse_clean / mse_adv if mse_adv != 0 else np.nan

    exceed_5 = np.mean(abs_adv_errors > 5.0) * 100.0
    exceed_10 = np.mean(abs_adv_errors > 10.0) * 100.0
    exceed_15 = np.mean(abs_adv_errors > 15.0) * 100.0

    return {
        "N": len(actuals),
        "Clean_MSE": mse_clean,
        "Clean_MAE": mae_clean,
        "Adv_MSE": mse_adv,
        "Adv_MAE": mae_adv,
        "RS": robustness_score,
        "Adv_Error_>5deg_%": exceed_5,
        "Adv_Error_>10deg_%": exceed_10,
        "Adv_Error_>15deg_%": exceed_15,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate_model_with_pgd(
    architecture,
    data_dir,
    weights_path,
    save_folder,
    ground_truth,
    strategy_short,
    epsilon=0.03,
    alpha=0.005,
    iterations=10,
):
    os.makedirs(save_folder, exist_ok=True)

    model = build_model(architecture)
    model.load_weights(weights_path)

    predictions = []
    adversarial_predictions = []
    actuals = []

    arch_file_prefix = "pilotnet" if architecture == "PilotNet" else "resnet"

    prediction_csv_path = os.path.join(
        save_folder,
        f"{arch_file_prefix}_{strategy_short}_pgd_sensitivity_eps{epsilon}_alpha{alpha}_iter{iterations}.csv"
    )

    with open(prediction_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Frame",
                "Ground_Truth",
                "Clean_Prediction",
                "Adversarial_Prediction",
                "Clean_Error",
                "Adversarial_Error",
                "Absolute_Adversarial_Error",
            ],
        )
        writer.writeheader()

        for frame_number in ground_truth.keys():
            image_name = f"frame_{frame_number}.jpg"
            image_path = os.path.join(data_dir, image_name)

            if not os.path.exists(image_path):
                continue

            processed_image = preprocess_image(image_path)
            actual_steering = ground_truth.get(frame_number)

            if actual_steering is None:
                continue

            clean_prediction_rad = model.predict(
                np.expand_dims(processed_image, axis=0),
                verbose=0
            )[0][0]

            adversarial_image = pgd_attack(
                model,
                processed_image,
                actual_steering,
                epsilon=epsilon,
                alpha=alpha,
                iterations=iterations,
            )

            if tf.is_tensor(adversarial_image):
                adversarial_image = adversarial_image.numpy()

            adversarial_prediction_rad = model.predict(
                np.expand_dims(adversarial_image, axis=0),
                verbose=0
            )[0][0]

            # Convert predictions from radians to degrees
            clean_prediction_deg = clean_prediction_rad * 180.0 / np.pi
            adversarial_prediction_deg = adversarial_prediction_rad * 180.0 / np.pi

            clean_error = clean_prediction_deg - actual_steering
            adversarial_error = adversarial_prediction_deg - actual_steering
            abs_adv_error = abs(adversarial_error)

            writer.writerow({
                "Frame": frame_number,
                "Ground_Truth": actual_steering,
                "Clean_Prediction": clean_prediction_deg,
                "Adversarial_Prediction": adversarial_prediction_deg,
                "Clean_Error": clean_error,
                "Adversarial_Error": adversarial_error,
                "Absolute_Adversarial_Error": abs_adv_error,
            })

            predictions.append(clean_prediction_deg)
            adversarial_predictions.append(adversarial_prediction_deg)
            actuals.append(actual_steering)

    metrics = compute_metrics(predictions, adversarial_predictions, actuals)

    summary_txt_path = os.path.join(
        save_folder,
        f"{arch_file_prefix}_{strategy_short}_pgd_sensitivity_summary_eps{epsilon}_alpha{alpha}_iter{iterations}.txt"
    )

    with open(summary_txt_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print(f"Completed {architecture} {strategy_short} | eps={epsilon} | alpha={alpha}")
    print(metrics)

    tf.keras.backend.clear_session()

    return metrics


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    DATA_DIR = str(TEST_IMAGE_DIR)
    GROUND_TRUTH_PATH = str(TEST_LABEL_CSV)

    OUTPUT_ROOT = str(RESULTS_DIR / "pgd_step_size_sensitivity_eps0.03")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    experiments = [
        # ====================================================
        # PilotNet
        # ====================================================
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
            "strategy": "Fine-tuned from U.S.-pretrained (frozen backbone)",
            "short_name": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_FT_US),
        },
        {
            "architecture": "PilotNet",
            "strategy": "Fine-tuned from Flipped-U.S.-pretrained (frozen backbone)",
            "short_name": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_FT_FLIPPED),
        },

        # ====================================================
        # ResNet-18
        # ====================================================
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
            "strategy": "Fine-tuned from U.S.-pretrained (frozen backbone)",
            "short_name": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_RESNET_FT_US),
        },
        {
            "architecture": "ResNet-18",
            "strategy": "Fine-tuned from Flipped-U.S.-pretrained (frozen backbone)",
            "short_name": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_RESNET_FT_FLIPPED),
        },
    ]

    summary_rows = []

    for exp in experiments:
        if not os.path.exists(exp["weights"]):
            print(f"WARNING: Missing weights file: {exp['weights']}")
            continue

        for alpha in SENSITIVITY_ALPHAS:
            print("=" * 90)
            print(
                f"Running PGD step-size sensitivity | "
                f"{exp['architecture']} | {exp['strategy']} | "
                f"epsilon={SENSITIVITY_EPSILON} | alpha={alpha} | iterations={ITERATIONS}"
            )
            print("Weights:", exp["weights"])

            arch_folder = "PilotNet" if exp["architecture"] == "PilotNet" else "ResNet18"

            save_folder = os.path.join(
                OUTPUT_ROOT,
                arch_folder,
                f"{exp['short_name']}_eps{SENSITIVITY_EPSILON}_alpha{alpha}_iter{ITERATIONS}"
            )

            metrics = evaluate_model_with_pgd(
                architecture=exp["architecture"],
                data_dir=DATA_DIR,
                weights_path=exp["weights"],
                save_folder=save_folder,
                ground_truth=ground_truth,
                strategy_short=exp["short_name"],
                epsilon=SENSITIVITY_EPSILON,
                alpha=alpha,
                iterations=ITERATIONS,
            )

            row = {
                "Architecture": exp["architecture"],
                "Strategy": exp["strategy"],
                "epsilon": SENSITIVITY_EPSILON,
                "alpha": alpha,
                "iterations": ITERATIONS,
                **metrics,
            }

            summary_rows.append(row)

    summary_csv_path = os.path.join(
        OUTPUT_ROOT,
        "pgd_step_size_sensitivity_summary.csv"
    )

    fieldnames = [
        "Architecture",
        "Strategy",
        "epsilon",
        "alpha",
        "iterations",
        "N",
        "Clean_MSE",
        "Clean_MAE",
        "Adv_MSE",
        "Adv_MAE",
        "RS",
        "Adv_Error_>5deg_%",
        "Adv_Error_>10deg_%",
        "Adv_Error_>15deg_%",
    ]

    with open(summary_csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 90)
    print("PGD step-size sensitivity analysis complete.")
    print(f"Summary saved to: {summary_csv_path}")
