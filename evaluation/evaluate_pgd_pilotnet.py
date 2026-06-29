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
from config import (
    TEST_IMAGE_DIR, TEST_LABEL_CSV, RESULTS_PGD_PILOTNET_DIR,
    WEIGHTS_PILOTNET_US, WEIGHTS_PILOTNET_FLIPPED,
    WEIGHTS_PILOTNET_FT_US, WEIGHTS_PILOTNET_FT_FLIPPED,
)



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

ITERATIONS = 10

PGD_SETTINGS = [
    {"epsilon": 0.01, "alpha": 0.002},
    {"epsilon": 0.03, "alpha": 0.005},
    {"epsilon": 0.05, "alpha": 0.01},
]



def pgd_attack(model, image, label_degrees, epsilon=0.01, alpha=0.002, iterations=10):
    """
    Corrected PGD attack.

    Model output: radians
    Ground truth in CSV: degrees
    PGD loss: radians
    Reported predictions: degrees
    """

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

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
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(label_tensor, prediction)


        gradients = tape.gradient(loss, adv_image)
        signed_grad = tf.sign(gradients)

        adv_image = adv_image + alpha * signed_grad

        perturbation = tf.clip_by_value(
            adv_image - image_tensor,
            -epsilon,
            epsilon
        )

        adv_image = image_tensor + perturbation
        adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    return adv_image



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
                        steering_angle = float(steering_angle)
                        ground_truth[image_index] = steering_angle

                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")

        else:
            for row in reader:
                if len(row) >= 2:
                    try:
                        image_index = int(
                            str(row[0])
                            .replace("frame_", "")
                            .replace(".jpg", "")
                            .split(".")[0]
                        )
                        steering_angle = float(row[1])
                        ground_truth[image_index] = steering_angle

                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")

    return ground_truth


def evaluate_model_with_pgd(
    data_dir,
    weights_path,
    save_folder,
    ground_truth,
    strategy="model",
    epsilon=0.01,
    alpha=0.002,
    iterations=10,
):
    os.makedirs(save_folder, exist_ok=True)

    model = PilotNet(input_shape=(66, 200, 3)).build_model()
    model.load_weights(weights_path)

    predictions = []
    adversarial_predictions = []
    actuals = []

    csv_path = os.path.join(
        save_folder,
        f"pilotnet_{strategy}_pgd_eps{epsilon}_alpha{alpha}_iter{iterations}.csv"
    )

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Frame",
                "Ground_Truth",
                "Clean_Prediction",
                "Adversarial_Prediction",
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

            clean_prediction_deg = clean_prediction_rad * 180.0 / np.pi
            adversarial_prediction_deg = adversarial_prediction_rad * 180.0 / np.pi

            writer.writerow({
                "Frame": frame_number,
                "Ground_Truth": actual_steering,
                "Clean_Prediction": clean_prediction_deg,
                "Adversarial_Prediction": adversarial_prediction_deg,
            })

            predictions.append(clean_prediction_deg)
            adversarial_predictions.append(adversarial_prediction_deg)
            actuals.append(actual_steering)

    predictions = np.array(predictions)
    adversarial_predictions = np.array(adversarial_predictions)
    actuals = np.array(actuals)

    clean_errors = predictions - actuals
    adversarial_errors = adversarial_predictions - actuals

    mse_clean = np.mean(clean_errors ** 2)
    mse_adv = np.mean(adversarial_errors ** 2)

    mae_clean = np.mean(np.abs(clean_errors))
    mae_adv = np.mean(np.abs(adversarial_errors))

    robustness_score = mse_clean / mse_adv if mse_adv != 0 else np.nan

    exceed_5 = np.mean(np.abs(adversarial_errors) > 5) * 100
    exceed_10 = np.mean(np.abs(adversarial_errors) > 10) * 100
    exceed_15 = np.mean(np.abs(adversarial_errors) > 15) * 100

    print(f"Clean MSE: {mse_clean}")
    print(f"Adversarial MSE: {mse_adv}")
    print(f"Clean MAE: {mae_clean}")
    print(f"Adversarial MAE: {mae_adv}")
    print(f"Robustness Score: {robustness_score}")
    print(f"Adv error >5 deg: {exceed_5}%")
    print(f"Adv error >10 deg: {exceed_10}%")
    print(f"Adv error >15 deg: {exceed_15}%")

    summary_path = os.path.join(
        save_folder,
        f"pilotnet_{strategy}_pgd_summary_eps{epsilon}_alpha{alpha}_iter{iterations}.txt"
    )

    with open(summary_path, "w") as f:
        f.write(f"MSE (Clean): {mse_clean}\n")
        f.write(f"MSE (Adversarial): {mse_adv}\n")
        f.write(f"MAE (Clean): {mae_clean}\n")
        f.write(f"MAE (Adversarial): {mae_adv}\n")
        f.write(f"Robustness Score: {robustness_score}\n")
        f.write(f"Adv error >5 deg (%): {exceed_5}\n")
        f.write(f"Adv error >10 deg (%): {exceed_10}\n")
        f.write(f"Adv error >15 deg (%): {exceed_15}\n")

    tf.keras.backend.clear_session()

    return predictions, adversarial_predictions, actuals


if __name__ == "__main__":

    DATA_DIR = str(TEST_IMAGE_DIR)
    GROUND_TRUTH_PATH = str(TEST_LABEL_CSV)

    OUTPUT_ROOT = str(RESULTS_PGD_PILOTNET_DIR)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    experiments = [
        {
            "strategy": "us_trained",
            "weights": str(WEIGHTS_PILOTNET_US),
        },
        {
            "strategy": "flipped_us_trained",
            "weights": str(WEIGHTS_PILOTNET_FLIPPED),
        },
        {
            "strategy": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_FT_US),
        },
        {
            "strategy": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_PILOTNET_FT_FLIPPED),
        },
    ]

    for exp in experiments:
        for setting in PGD_SETTINGS:
            epsilon = setting["epsilon"]
            alpha = setting["alpha"]

            print("=" * 70)
            print(
                f"Running PilotNet {exp['strategy']} | "
                f"eps={epsilon} | alpha={alpha} | iter={ITERATIONS}"
            )
            print("Weights:", exp["weights"])

            if not os.path.exists(exp["weights"]):
                print(f"WARNING: Missing weights file: {exp['weights']}")
                continue

            save_folder = os.path.join(
                OUTPUT_ROOT,
                f"pilotnet_{exp['strategy']}_pgd_eps{epsilon}_alpha{alpha}_iter{ITERATIONS}"
            )

            evaluate_model_with_pgd(
                data_dir=DATA_DIR,
                weights_path=exp["weights"],
                save_folder=save_folder,
                ground_truth=ground_truth,
                strategy=exp["strategy"],
                epsilon=epsilon,
                alpha=alpha,
                iterations=ITERATIONS,
            )
