"""
Transfer-based black-box robustness evaluation.

This script evaluates cross-architecture adversarial transferability for
the study:
"Domain Adaptation Affects Adversarial Robustness in Autonomous Steering Models"

Adversarial examples are generated using one architecture as the surrogate
model and evaluated on the other architecture as the target model, without
using target-model gradients.

Expected project structure and all data/weight paths are defined in config.py.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import (
    TEST_IMAGE_DIR,
    TEST_LABEL_CSV,
    RESULTS_DIR,
    RESULTS_TRANSFER_BLACKBOX_DIR,
    WEIGHTS_PILOTNET_US,
    WEIGHTS_PILOTNET_FLIPPED,
    WEIGHTS_PILOTNET_PARTIAL_FT_US,
    WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED,
    WEIGHTS_RESNET_US,
    WEIGHTS_RESNET_FLIPPED,
    WEIGHTS_RESNET_PARTIAL_FT_US,
    WEIGHTS_RESNET_PARTIAL_FT_FLIPPED,
)

from processing2 import preprocess_image
from pilotnet import PilotNet
from ResNet_Tensorflow import ResNet18Steering


# ============================================================
# Settings
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

ITERATIONS = 10

FGSM_EPSILONS = [0.01, 0.03, 0.05]

PGD_SETTINGS = {
    0.01: 0.002,
    0.03: 0.005,
    0.05: 0.010,
}


# ============================================================
# Attack functions
# ============================================================

def fgsm_attack(model, image, label_degrees, epsilon=0.01):
    """
    Generate an FGSM adversarial example using the surrogate model.

    Model output: radians
    Ground-truth label: degrees
    Attack loss: radians
    Image range: [0, 1]
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(tf.expand_dims(image_tensor, axis=0), training=False)
        loss = tf.keras.losses.MeanSquaredError()(label_tensor, prediction)

    gradients = tape.gradient(loss, image_tensor)
    signed_grad = tf.sign(gradients)

    adversarial_image = image_tensor + epsilon * signed_grad
    adversarial_image = tf.clip_by_value(adversarial_image, 0.0, 1.0)

    return adversarial_image


def pgd_attack(model, image, label_degrees, epsilon=0.01, alpha=0.002, iterations=10):
    """
    Generate a PGD adversarial example using the surrogate model.

    Model output: radians
    Ground-truth label: degrees
    Attack loss: radians
    Image range: [0, 1]
    """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

    # Random initialization within the allowed perturbation region.
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

        perturbation = tf.clip_by_value(adv_image - image_tensor, -epsilon, epsilon)
        adv_image = image_tensor + perturbation
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    return adv_image


# ============================================================
# Utility functions
# ============================================================

def load_ground_truth(csv_path):
    """
    Load ground-truth steering angles from CSV.

    Supported formats:
    1. Two-column CSV: frame_name, steering_angle
    2. Space-separated single-column rows: frame_name steering_angle
    """
    ground_truth = {}

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader)

        if len(headers) == 1:
            for row in reader:
                if not row:
                    continue

                parts = row[0].split()
                if len(parts) < 2:
                    continue

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
                except ValueError:
                    continue

        else:
            for row in reader:
                if len(row) < 2:
                    continue

                try:
                    image_index = int(
                        str(row[0])
                        .replace("frame_", "")
                        .replace(".jpg", "")
                        .split(".")[0]
                    )
                    steering_angle = float(row[1])
                    ground_truth[image_index] = steering_angle
                except ValueError:
                    continue

    if not ground_truth:
        raise ValueError(f"No valid ground-truth labels loaded from: {csv_path}")

    return ground_truth


def build_model(architecture):
    """Build model architecture."""
    if architecture == "PilotNet":
        return PilotNet(input_shape=(66, 200, 3)).build_model()

    if architecture == "ResNet":
        return ResNet18Steering(input_shape=(66, 200, 3)).build_model()

    raise ValueError(f"Unknown architecture: {architecture}")


def predict_degrees(model, image):
    """
    Predict steering angle in degrees.

    The model output is assumed to be in radians, so predictions are converted
    back to degrees for metric reporting.
    """
    pred_rad = model.predict(np.expand_dims(image, axis=0), verbose=0)[0][0]
    return float(pred_rad * 180.0 / np.pi)


def calculate_metrics(actuals, clean_predictions, adversarial_predictions):
    """Calculate degree-based clean and transferred adversarial metrics."""
    actuals = np.array(actuals, dtype=np.float32)
    clean_predictions = np.array(clean_predictions, dtype=np.float32)
    adversarial_predictions = np.array(adversarial_predictions, dtype=np.float32)

    clean_errors = clean_predictions - actuals
    adversarial_errors = adversarial_predictions - actuals

    mse_clean = float(np.mean(clean_errors ** 2))
    mse_adv = float(np.mean(adversarial_errors ** 2))

    mae_clean = float(np.mean(np.abs(clean_errors)))
    mae_adv = float(np.mean(np.abs(adversarial_errors)))

    robustness_score = float(mse_clean / mse_adv) if mse_adv != 0 else np.nan

    exceed_5 = float(np.mean(np.abs(adversarial_errors) > 5) * 100)
    exceed_10 = float(np.mean(np.abs(adversarial_errors) > 10) * 100)
    exceed_15 = float(np.mean(np.abs(adversarial_errors) > 15) * 100)

    return {
        "Clean_MSE_Target": mse_clean,
        "Transfer_Adv_MSE_Target": mse_adv,
        "Clean_MAE_Target": mae_clean,
        "Transfer_Adv_MAE_Target": mae_adv,
        "Robustness_Score": robustness_score,
        "Adv_error_gt_5_deg_percent": exceed_5,
        "Adv_error_gt_10_deg_percent": exceed_10,
        "Adv_error_gt_15_deg_percent": exceed_15,
    }


# ============================================================
# Transfer-based black-box evaluation
# ============================================================

def evaluate_transfer_blackbox(
    surrogate_architecture,
    target_architecture,
    surrogate_weights_path,
    target_weights_path,
    data_dir,
    ground_truth,
    save_folder,
    condition_name,
    attack_name="FGSM",
    epsilon=0.03,
    alpha=0.005,
    iterations=10,
):
    """
    Run transfer-based black-box evaluation.

    The attack is generated using the surrogate model.
    The adversarial image is evaluated using the target model.
    Target-model gradients are not used to generate the attack.
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    surrogate_weights_path = Path(surrogate_weights_path)
    target_weights_path = Path(target_weights_path)
    data_dir = Path(data_dir)

    if not surrogate_weights_path.exists():
        raise FileNotFoundError(f"Missing surrogate weights: {surrogate_weights_path}")

    if not target_weights_path.exists():
        raise FileNotFoundError(f"Missing target weights: {target_weights_path}")

    if not data_dir.exists():
        raise FileNotFoundError(f"Missing test image directory: {data_dir}")

    print("=" * 80)
    print(f"Transfer black-box: {surrogate_architecture} -> {target_architecture}")
    print(f"Condition: {condition_name}")
    print(f"Attack: {attack_name}, epsilon={epsilon}, alpha={alpha}, iterations={iterations}")

    surrogate_model = build_model(surrogate_architecture)
    surrogate_model.load_weights(str(surrogate_weights_path))

    target_model = build_model(target_architecture)
    target_model.load_weights(str(target_weights_path))

    clean_predictions = []
    adversarial_predictions = []
    actuals = []

    direction_name = f"{surrogate_architecture}_to_{target_architecture}"

    csv_path = save_folder / (
        f"transfer_{attack_name.lower()}_{direction_name}_{condition_name}_eps{epsilon}.csv"
    )

    missing_images = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Frame",
                "Ground_Truth",
                "Clean_Target_Prediction",
                "Transfer_Adversarial_Target_Prediction",
            ],
        )
        writer.writeheader()

        for frame_number in sorted(ground_truth.keys()):
            image_name = f"frame_{frame_number}.jpg"
            image_path = data_dir / image_name

            if not image_path.exists():
                missing_images += 1
                continue

            processed_image = preprocess_image(str(image_path))
            actual_steering = ground_truth.get(frame_number)

            clean_prediction_deg = predict_degrees(target_model, processed_image)

            if attack_name.upper() == "FGSM":
                adversarial_image = fgsm_attack(
                    model=surrogate_model,
                    image=processed_image,
                    label_degrees=actual_steering,
                    epsilon=epsilon,
                )
            elif attack_name.upper() == "PGD":
                adversarial_image = pgd_attack(
                    model=surrogate_model,
                    image=processed_image,
                    label_degrees=actual_steering,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=iterations,
                )
            else:
                raise ValueError(f"Unknown attack name: {attack_name}")

            if tf.is_tensor(adversarial_image):
                adversarial_image = adversarial_image.numpy()

            adversarial_prediction_deg = predict_degrees(target_model, adversarial_image)

            writer.writerow({
                "Frame": frame_number,
                "Ground_Truth": actual_steering,
                "Clean_Target_Prediction": clean_prediction_deg,
                "Transfer_Adversarial_Target_Prediction": adversarial_prediction_deg,
            })

            clean_predictions.append(clean_prediction_deg)
            adversarial_predictions.append(adversarial_prediction_deg)
            actuals.append(actual_steering)

    if not actuals:
        raise RuntimeError(
            "No images were evaluated. Check TEST_IMAGE_DIR, TEST_LABEL_CSV, and frame naming."
        )

    metrics = calculate_metrics(actuals, clean_predictions, adversarial_predictions)

    summary_path = save_folder / (
        f"transfer_{attack_name.lower()}_{direction_name}_{condition_name}_summary_eps{epsilon}.txt"
    )

    with open(summary_path, "w") as f:
        f.write(f"Attack: {attack_name}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Surrogate: {surrogate_architecture}\n")
        f.write(f"Target: {target_architecture}\n")
        f.write(f"Direction: {surrogate_architecture} -> {target_architecture}\n")
        f.write(f"Condition: {condition_name}\n")
        f.write(f"Evaluated images: {len(actuals)}\n")
        f.write(f"Missing images: {missing_images}\n")
        f.write(f"Clean MSE Target: {metrics['Clean_MSE_Target']}\n")
        f.write(f"Transfer Adv MSE Target: {metrics['Transfer_Adv_MSE_Target']}\n")
        f.write(f"Clean MAE Target: {metrics['Clean_MAE_Target']}\n")
        f.write(f"Transfer Adv MAE Target: {metrics['Transfer_Adv_MAE_Target']}\n")
        f.write(f"Robustness Score: {metrics['Robustness_Score']}\n")
        f.write(f"Adv error >5 deg (%): {metrics['Adv_error_gt_5_deg_percent']}\n")
        f.write(f"Adv error >10 deg (%): {metrics['Adv_error_gt_10_deg_percent']}\n")
        f.write(f"Adv error >15 deg (%): {metrics['Adv_error_gt_15_deg_percent']}\n")
        f.write(f"Prediction CSV: {csv_path}\n")

    result_row = {
        "Attack": attack_name.upper(),
        "Epsilon": epsilon,
        "Alpha": alpha if attack_name.upper() == "PGD" else "",
        "Iterations": iterations if attack_name.upper() == "PGD" else "",
        "Surrogate": surrogate_architecture,
        "Target": target_architecture,
        "Direction": f"{surrogate_architecture} -> {target_architecture}",
        "Condition": condition_name,
        "Evaluated_Images": len(actuals),
        "Missing_Images": missing_images,
        **metrics,
        "Prediction_CSV": str(csv_path),
        "Summary_TXT": str(summary_path),
    }

    tf.keras.backend.clear_session()
    return result_row


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run transfer-based black-box attacks between PilotNet and ResNet-18."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.03,
        choices=[0.01, 0.03, 0.05],
        help="Single epsilon value to run. Default: 0.03",
    )
    parser.add_argument(
        "--all-eps",
        action="store_true",
        help="Run all epsilon values: 0.01, 0.03, 0.05",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="both",
        choices=["fgsm", "pgd", "both"],
        help="Which attack to run: fgsm, pgd, or both.",
    )
    args = parser.parse_args()

    output_root = RESULTS_TRANSFER_BLACKBOX_DIR
    output_root.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth(TEST_LABEL_CSV)

    conditions = [
        {
            "strategy": "pretrained",
            "pilotnet_weights": WEIGHTS_PILOTNET_US,
            "resnet_weights": WEIGHTS_RESNET_US,
        },
        {
            "strategy": "flipped",
            "pilotnet_weights": WEIGHTS_PILOTNET_FLIPPED,
            "resnet_weights": WEIGHTS_RESNET_FLIPPED,
        },
        {
            "strategy": "finetuned",
            "pilotnet_weights": WEIGHTS_PILOTNET_PARTIAL_FT_US,
            "resnet_weights": WEIGHTS_RESNET_PARTIAL_FT_US,
        },
        {
            "strategy": "finetunedflipped",
            "pilotnet_weights": WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED,
            "resnet_weights": WEIGHTS_RESNET_PARTIAL_FT_FLIPPED,
        },
    ]

    epsilons = FGSM_EPSILONS if args.all_eps else [args.eps]

    if args.attack == "both":
        attacks = ["FGSM", "PGD"]
    elif args.attack == "fgsm":
        attacks = ["FGSM"]
    else:
        attacks = ["PGD"]

    all_rows = []

    for condition in conditions:
        strategy = condition["strategy"]
        pilotnet_weights = condition["pilotnet_weights"]
        resnet_weights = condition["resnet_weights"]

        for epsilon in epsilons:
            alpha = PGD_SETTINGS[epsilon]

            for attack_name in attacks:
                save_folder_1 = output_root / (
                    f"{attack_name.lower()}_PilotNet_to_ResNet_{strategy}_eps{epsilon}"
                )

                row_1 = evaluate_transfer_blackbox(
                    surrogate_architecture="PilotNet",
                    target_architecture="ResNet",
                    surrogate_weights_path=pilotnet_weights,
                    target_weights_path=resnet_weights,
                    data_dir=TEST_IMAGE_DIR,
                    ground_truth=ground_truth,
                    save_folder=save_folder_1,
                    condition_name=strategy,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=ITERATIONS,
                )
                all_rows.append(row_1)

                save_folder_2 = output_root / (
                    f"{attack_name.lower()}_ResNet_to_PilotNet_{strategy}_eps{epsilon}"
                )

                row_2 = evaluate_transfer_blackbox(
                    surrogate_architecture="ResNet",
                    target_architecture="PilotNet",
                    surrogate_weights_path=resnet_weights,
                    target_weights_path=pilotnet_weights,
                    data_dir=TEST_IMAGE_DIR,
                    ground_truth=ground_truth,
                    save_folder=save_folder_2,
                    condition_name=strategy,
                    attack_name=attack_name,
                    epsilon=epsilon,
                    alpha=alpha,
                    iterations=ITERATIONS,
                )
                all_rows.append(row_2)

    combined_summary_path = output_root / "transfer_blackbox_summary_all.csv"

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(combined_summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        print("=" * 80)
        print("All transfer black-box experiments completed.")
        print("Combined summary saved to:", combined_summary_path)
    else:
        print("No experiments were completed. Please check the config paths.")


if __name__ == "__main__":
    main()