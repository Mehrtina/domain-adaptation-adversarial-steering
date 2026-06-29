import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from data.dataloader import DataLoader
from models.pilotnet import PilotNet
from config import (
    DATA_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    L2_REG_CONST, LOG_DIR_TRAIN_PILOTNET, WEIGHTS_PILOTNET_US,
)

import os

os.makedirs(LOG_DIR_TRAIN_PILOTNET, exist_ok=True)


def create_custom_loss(model, l2_reg_const):
    def custom_loss(y_true, y_pred):
        """Custom loss function with L2 regularization."""
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        return mse + l2_reg_const * l2_loss

    return custom_loss


def plot_loss(history):
    """Plot the training and validation loss."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train():
    train_loader = DataLoader(DATA_DIR, BATCH_SIZE, is_training=True)
    val_loader = DataLoader(DATA_DIR, BATCH_SIZE, is_training=False)

    pilotnet = PilotNet(input_shape=(66, 200, 3))
    model = pilotnet.build_model()

    loss_function = create_custom_loss(model, L2_REG_CONST)

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_function)

    tensorboard_callback = TensorBoard(log_dir=str(LOG_DIR_TRAIN_PILOTNET), histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(
        filepath=str(WEIGHTS_PILOTNET_US),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )

    history = model.fit(
        train_loader,
        epochs=NUM_EPOCHS,
        validation_data=val_loader,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    plot_loss(history)


if __name__ == "__main__":
    train()
