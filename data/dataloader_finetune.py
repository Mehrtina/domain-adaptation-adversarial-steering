import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from preprocessing.preprocess_australian import preprocess_image


class DataLoader(Sequence):
    """Data loader for Australian fine-tuning dataset.
    """

    def __init__(self, data_dir, batch_size, is_training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.samples = self._load_samples()
        print(f"Total samples: {len(self.samples)}")
        self.on_epoch_end()

    def _load_samples(self):
        samples = []
        label_file = os.path.join(self.data_dir, "totalDATA.txt")
        with open(label_file, "r") as f:
            for line in f:
                frame_number, angle = line.strip().split()
                img_path = f"frame_{frame_number}.jpg"
                angle_in_radians = float(angle) * np.pi / 180  # degrees → radians
                samples.append((os.path.join(self.data_dir, img_path), angle_in_radians))
        return samples

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        X, y = [], []
        for img_path, angle in batch_samples:
            img = preprocess_image(img_path)
            X.append(img)
            y.append(angle)
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.samples)
