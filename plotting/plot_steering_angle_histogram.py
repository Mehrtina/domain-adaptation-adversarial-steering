import os
import sys
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from config import TEST_LABEL_CSV, FIGURES_DIR

GROUND_TRUTH_PATH = str(TEST_LABEL_CSV)

OUTPUT_PNG = str(FIGURES_DIR / "test_steering_angle_histogram.png")
OUTPUT_CSV = str(FIGURES_DIR / "test_steering_angle_summary.csv")


def load_steering_angles(csv_path):
    angles = []

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
                    try:
                        steering_angle = float(parts[1])
                        angles.append(steering_angle)
                    except ValueError:
                        continue

        else:
            for row in reader:
                if len(row) >= 2:
                    try:
                        steering_angle = float(row[1])
                        angles.append(steering_angle)
                    except ValueError:
                        continue

    return np.array(angles, dtype=float)


os.makedirs(FIGURES_DIR, exist_ok=True)

angles = load_steering_angles(GROUND_TRUTH_PATH)

print("Number of steering labels:", len(angles))
print("Mean:", np.mean(angles))
print("Median:", np.median(angles))
print("Std:", np.std(angles))
print("Min:", np.min(angles))
print("Max:", np.max(angles))
print("5th percentile:", np.percentile(angles, 5))
print("95th percentile:", np.percentile(angles, 95))

# Save numerical summary
summary = {
    "N": len(angles),
    "Mean": np.mean(angles),
    "Median": np.median(angles),
    "Std": np.std(angles),
    "Min": np.min(angles),
    "Max": np.max(angles),
    "P5": np.percentile(angles, 5),
    "P25": np.percentile(angles, 25),
    "P75": np.percentile(angles, 75),
    "P95": np.percentile(angles, 95),
}

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

# Plot histogram
plt.figure(figsize=(7, 5))
plt.hist(angles, bins=50, edgecolor="black")
plt.xlabel("Ground-truth steering angle (degrees)")
plt.ylabel("Number of test images")
plt.title("Australian held-out test set steering-angle distribution")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300)
plt.close()

print("Saved:", OUTPUT_PNG)
print("Saved:", OUTPUT_CSV)
