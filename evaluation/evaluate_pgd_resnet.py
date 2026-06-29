import os
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocessing.preprocess_australian import preprocess_image
from models.resnet18 import ResNet18Steering
from config import (
    TEST_IMAGE_DIR, TEST_LABEL_CSV, RESULTS_PGD_RESNET_DIR,
    WEIGHTS_RESNET_US, WEIGHTS_RESNET_FLIPPED,
    WEIGHTS_RESNET_FT_US, WEIGHTS_RESNET_FT_FLIPPED,
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

    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    
    label_radians = label_degrees * np.pi / 180.0
    label_tensor = tf.convert_to_tensor([[label_radians]], dtype=tf.float32)

  
    adv_image = image_tensor + tf.random.uniform(image_tensor.shape, -epsilon/2, epsilon/2, dtype=tf.float32)
    adv_image = tf.clip_by_value(adv_image, 0.0, 1.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(adv_image)
            prediction = model(tf.expand_dims(adv_image, 0), training=False)
            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(label_tensor, prediction)

        gradients = tape.gradient(loss, adv_image)

        signed_grad = tf.sign(gradients)
        adv_image = adv_image + alpha * signed_grad

        perturbation = tf.clip_by_value(adv_image - image_tensor, -epsilon, epsilon)
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
                if row:
                    parts = row[0].split()
                    if len(parts) >= 2:
                        image_name, steering_angle = parts[0], parts[1]
                        try:
                            image_index = int(image_name.split('.')[0])
                            steering_angle = float(steering_angle)
                            ground_truth[image_index] = steering_angle
                        except ValueError as e:
                            print(f"Skipping row due to error: {e}")
        else:
            for row in reader:
                if len(row) >= 2:
                    try:
                        image_index = int(row[0])
                        steering_angle = float(row[1])
                        ground_truth[image_index] = steering_angle
                    except ValueError as e:
                        print(f"Skipping row due to error: {e}")
                else:
                    print(f"Skipping row due to insufficient data: {row}")

    return ground_truth


def evaluate_model_with_pgd(data_dir, weights_path, save_folder, ground_truth, strategy="model", epsilon=0.01, alpha=0.002, iterations=10):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    model = ResNet18Steering(input_shape=(66, 200, 3)).build_model()
    model.load_weights(weights_path)
    predictions, adversarial_predictions, actuals = [], [], []

    csv_path = os.path.join(save_folder, f"resnet_{strategy}_pgd_eps{epsilon}_alpha{alpha}_iter{iterations}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Frame', 'Ground_Truth', 'Clean_Prediction', 'Adversarial_Prediction'])
        writer.writeheader()

        for frame_number in ground_truth.keys():
            image_name = f"frame_{frame_number}.jpg"
            image_path = os.path.join(data_dir, image_name)
            if not os.path.exists(image_path):
                continue

            processed_image = preprocess_image(image_path)
            clean_prediction = model.predict(np.expand_dims(processed_image, axis=0))[0][0]
            actual_steering = ground_truth.get(frame_number)
            if actual_steering is None:
                continue

            adversarial_image = pgd_attack(model, processed_image, actual_steering, epsilon=epsilon, alpha=alpha, iterations=iterations)
            if tf.is_tensor(adversarial_image):
                adversarial_image = adversarial_image.numpy()
            adversarial_prediction = model.predict(np.expand_dims(adversarial_image, axis=0))[0][0]

            writer.writerow({
                'Frame': frame_number,
                'Ground_Truth': actual_steering,
                'Clean_Prediction': clean_prediction * 180.0 / np.pi,
                'Adversarial_Prediction': adversarial_prediction * 180.0 / np.pi
            })

            predictions.append(clean_prediction * 180.0 / np.pi)
            adversarial_predictions.append(adversarial_prediction * 180.0 / np.pi)
            actuals.append(actual_steering)

    mse_clean = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    mse_adv = np.mean((np.array(adversarial_predictions) - np.array(actuals)) ** 2)
    robustness_score = mse_clean / mse_adv

    print(f"Clean MSE: {mse_clean}")
    print(f"Adversarial MSE: {mse_adv}")
    print(f"Robustness Score: {robustness_score}")

    with open(os.path.join(save_folder, f"resnet_{strategy}_pgd_summary_eps{epsilon}_alpha{alpha}_iter{iterations}.txt"), "w") as f:
        f.write(f"MSE (Clean): {mse_clean}\n")
        f.write(f"MSE (Adversarial): {mse_adv}\n")
        f.write(f"Robustness Score: {robustness_score}\n")

    return np.array(predictions), np.array(adversarial_predictions), np.array(actuals)


if __name__ == "__main__":

    DATA_DIR = str(TEST_IMAGE_DIR)
    GROUND_TRUTH_PATH = str(TEST_LABEL_CSV)

    OUTPUT_ROOT = str(RESULTS_PGD_RESNET_DIR)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    experiments = [
        {
            "strategy": "us_trained",
            "weights": str(WEIGHTS_RESNET_US),
        },
        {
            "strategy": "flipped_us_trained",
            "weights": str(WEIGHTS_RESNET_FLIPPED),
        },
        {
            "strategy": "finetuned_us_pretrained",
            "weights": str(WEIGHTS_RESNET_FT_US),
        },
        {
            "strategy": "finetuned_flipped_us_pretrained",
            "weights": str(WEIGHTS_RESNET_FT_FLIPPED),
        },
    ]

    for exp in experiments:
        for setting in PGD_SETTINGS:
            epsilon = setting["epsilon"]
            alpha = setting["alpha"]

            print("=" * 70)
            print(
                f"Running ResNet {exp['strategy']} | "
                f"eps={epsilon} | alpha={alpha} | iter={ITERATIONS}"
            )
            print("Weights:", exp["weights"])

            if not os.path.exists(exp["weights"]):
                print(f"WARNING: Missing weights file: {exp['weights']}")
                continue

            save_folder = os.path.join(
                OUTPUT_ROOT,
                f"resnet_{exp['strategy']}_pgd_eps{epsilon}_alpha{alpha}_iter{ITERATIONS}"
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
