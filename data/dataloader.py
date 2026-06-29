import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from preprocessing.preprocess_us_training import preprocess_image

class DataLoader(Sequence):
    """Data loader for PilotNet dataset."""

    def __init__(self, data_dir, batch_size, is_training=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.is_training = is_training
        self.samples = self._load_samples()
        self.on_epoch_end()

    def _load_samples(self):
        samples = []
        with open(os.path.join(self.data_dir, "data.txt"), "r") as f:
            for line in f:
                img_path, angle = line.strip().split()
                angle_in_radians = float(angle) * np.pi / 180  # Convert degrees to radians
                #angle_in_degrees = float(angle)
                samples.append((os.path.join(self.data_dir, img_path), angle_in_radians))
        return samples



    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_samples = self.samples[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        X, y = [], []
        for sample in batch_samples:
            img_path, angle = sample
            #print(f"Loading image from: {img_path}")  # Debugging print
            img = preprocess_image(img_path)
                
             
            X.append(img)
            y.append(angle)
        return np.array(X), np.array(y)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.samples)

