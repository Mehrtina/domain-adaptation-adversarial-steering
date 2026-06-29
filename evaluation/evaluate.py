"""
Clean evaluation for PilotNet and ResNet-18 steering models.

This script evaluates clean, non-adversarial steering predictions on the test set.
All reported metrics are computed in degrees.

"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import ( 
    TEST_IMAGE_DIR,
    TEST_LABEL_CSV,
    RESULTS_DIR,
    WEIGHTS_PILOTNET_US,
    WEIGHTS_PILOTNET_FLIPPED,
    WEIGHTS_PILOTNET_PARTIAL_FT_US,
    WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED,
    WEIGHTS_RESNET_US,
    WEIGHTS_RESNET_FLIPPED,
    WEIGHTS_RESNET_PARTIAL_FT_US,
    WEIGHTS_RESNET_PARTIAL_FT_FLIPPED,
)
from preprocessing.preprocess_australian import preprocess_image  
from models.pilotnet import PilotNet  
from models.resnet18 import ResNet18Steering  



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



EXPERIMENTS: List[Dict[str, Any]] = [
    # --------------------------------------------------------
    # PilotNet
    # --------------------------------------------------------
    {
        "architecture": "PilotNet",
        "strategy": "us_trained",
        "weights": WEIGHTS_PILOTNET_US,
    },
    {
        "architecture": "PilotNet",
        "strategy": "flipped_us_trained",
        "weights": WEIGHTS_PILOTNET_FLIPPED,
    },
    {
        "architecture": "PilotNet",
        "strategy": "finetuned_us_pretrained",
        "weights": WEIGHTS_PILOTNET_PARTIAL_FT_US,
    },
    {
        "architecture": "PilotNet",
        "strategy": "finetuned_flipped_us_pretrained",
        "weights": WEIGHTS_PILOTNET_PARTIAL_FT_FLIPPED,
    },

    # --------------------------------------------------------
    # ResNet-18
    # --------------------------------------------------------
    {
        "architecture": "ResNet-18",
        "strategy": "us_trained",
        "weights": WEIGHTS_RESNET_US,
    },
    {
        "architecture": "ResNet-18",
        "strategy": "flipped_us_trained",
        "weights": WEIGHTS_RESNET_FLIPPED,
    },
    {
        "architecture": "ResNet-18",
        "strategy": "finetuned_us_pretrained",
        "weights": WEIGHTS_RESNET_PARTIAL_FT_US,
    },
    {
        "architecture": "ResNet-18",
        "strategy": "finetuned_flipped_us_pretrained",
        "weights": WEIGHTS_RESNET_PARTIAL_FT_FLIPPED,
    },
]


def parse_frame_index(value: str) -> int:
  
    value = str(value).strip()
    value = value.replace("frame_", "")
    value = value.replace(".jpg", "")
    value = value.replace(".jpeg", "")
    value = value.replace(".png", "")
    return int(value)


def load_ground_truth(csv_path: Path) -> Dict[int, float]:
    
    
    ground_truth: Dict[int, float] = {}

    with open(csv_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        print(f"CSV headers: {headers}")

        for row in reader:
            if len(row) < 2:
                continue

            try:
                frame_index = parse_frame_index(row[0])
                steering_angle_deg = float(row[1])
                ground_truth[frame_index] = steering_angle_deg
            except ValueError:
                print(f"Skipping invalid row: {row}")

    return ground_truth


def build_model(architecture: str):
   
    if architecture == "PilotNet":
        return PilotNet(input_shape=(66, 200, 3)).build_model()

    if architecture == "ResNet-18":
        return ResNet18Steering(input_shape=(66, 200, 3)).build_model()

    raise ValueError(f"Unsupported architecture: {architecture}")


def evaluate_clean_model(
    architecture: str,
    strategy: str,
    weights_path: Path,
    test_image_dir: Path,
    ground_truth: Dict[int, float],
    output_dir: Path,
) -> Dict[str, float]:

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(architecture)
    model.load_weights(str(weights_path))

    predictions_deg = []
    actuals_deg = []

    output_csv = output_dir / f"{architecture.lower().replace('-', '')}_{strategy}_clean_predictions.csv"

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "Frame",
                "Architecture",
                "Strategy",
                "Ground_Truth_Degrees",
                "Clean_Prediction_Degrees",
                "Absolute_Error_Degrees",
            ],
        )
        writer.writeheader()

        for frame_index in sorted(ground_truth.keys()):
            image_path = test_image_dir / f"frame_{frame_index}.jpg"

            if not image_path.exists():
                continue

            processed_image = preprocess_image(str(image_path))

            prediction_rad = model.predict(
                np.expand_dims(processed_image, axis=0),
                verbose=0,
            )[0][0]

            prediction_deg = float(prediction_rad * 180.0 / np.pi)
            actual_deg = float(ground_truth[frame_index])
            absolute_error_deg = abs(prediction_deg - actual_deg)

            predictions_deg.append(prediction_deg)
            actuals_deg.append(actual_deg)

            writer.writerow({
                "Frame": frame_index,
                "Architecture": architecture,
                "Strategy": strategy,
                "Ground_Truth_Degrees": actual_deg,
                "Clean_Prediction_Degrees": prediction_deg,
                "Absolute_Error_Degrees": absolute_error_deg,
            })

    predictions_deg = np.array(predictions_deg, dtype=np.float64)
    actuals_deg = np.array(actuals_deg, dtype=np.float64)

    if len(actuals_deg) == 0:
        raise RuntimeError(
            f"No images were evaluated for {architecture} / {strategy}. "
            f"Check TEST_IMAGE_DIR and TEST_LABEL_CSV."
        )

    errors = predictions_deg - actuals_deg

    clean_mse = float(np.mean(errors ** 2))
    clean_mae = float(np.mean(np.abs(errors)))
    clean_rmse = float(np.sqrt(clean_mse))

    summary = {
        "Architecture": architecture,
        "Strategy": strategy,
        "Num_Evaluated_Images": int(len(actuals_deg)),
        "Clean_MSE_Degrees": clean_mse,
        "Clean_MAE_Degrees": clean_mae,
        "Clean_RMSE_Degrees": clean_rmse,
    }

    summary_txt = output_dir / f"{architecture.lower().replace('-', '')}_{strategy}_clean_summary.txt"

    with open(summary_txt, mode="w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")

    print("=" * 70)
    print(f"Architecture: {architecture}")
    print(f"Strategy: {strategy}")
    print(f"Evaluated images: {len(actuals_deg)}")
    print(f"Clean MSE: {clean_mse:.6f}")
    print(f"Clean MAE: {clean_mae:.6f}")
    print(f"Clean RMSE: {clean_rmse:.6f}")
    print(f"Predictions saved to: {output_csv}")
    print("=" * 70)

    return summary


def save_master_summary(summaries: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save all clean-evaluation summaries into one CSV file.
    """
    if not summaries:
        print("No summaries to save.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Architecture",
        "Strategy",
        "Num_Evaluated_Images",
        "Clean_MSE_Degrees",
        "Clean_MAE_Degrees",
        "Clean_RMSE_Degrees",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)

    print(f"Master clean-evaluation summary saved to: {output_path}")



def main() -> None:
    test_image_dir = Path(TEST_IMAGE_DIR)
    test_label_csv = Path(TEST_LABEL_CSV)
    output_root = Path(RESULTS_DIR) / "clean_evaluation"

    if not test_image_dir.exists():
        raise FileNotFoundError(f"TEST_IMAGE_DIR does not exist: {test_image_dir}")

    if not test_label_csv.exists():
        raise FileNotFoundError(f"TEST_LABEL_CSV does not exist: {test_label_csv}")

    ground_truth = load_ground_truth(test_label_csv)

    summaries = []

    for experiment in EXPERIMENTS:
        architecture = experiment["architecture"]
        strategy = experiment["strategy"]
        weights_path = Path(experiment["weights"])

        if not weights_path.exists():
            print(
                f"WARNING: Missing weights for {architecture} / {strategy}: "
                f"{weights_path}. Skipping."
            )
            continue

        experiment_output_dir = output_root / architecture.lower().replace("-", "") / strategy

        summary = evaluate_clean_model(
            architecture=architecture,
            strategy=strategy,
            weights_path=weights_path,
            test_image_dir=test_image_dir,
            ground_truth=ground_truth,
            output_dir=experiment_output_dir,
        )

        summaries.append(summary)

    save_master_summary(
        summaries=summaries,
        output_path=output_root / "clean_evaluation_summary.csv",
    )


if __name__ == "__main__":
    main()