"""
Fine-tune U.S. pretrained (or flipped U.S. pretrained) PilotNet on Australian data.
Freeze strategy: all layers frozen except the final output layer (head-only).

"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from data.dataloader_finetune import DataLoader
from models.pilotnet import PilotNet
from config import (
    FINETUNE_DATA_DIR, LOGS_DIR,
    WEIGHTS_PILOTNET_US, WEIGHTS_PILOTNET_FT_HEAD_ONLY_US, BATCH_SIZE, NUM_EPOCHS_FT, LEARNING_RATE,
)

FINETUNE_DATA_DIR = Path(FINETUNE_DATA_DIR)
LOGS_DIR = Path(LOGS_DIR)

MODEL_CHECKPOINT_IN  = Path(WEIGHTS_PILOTNET_US)
LOG_DIR              = LOGS_DIR / "logs_frozen_backbone_finetuned_pilotnet"
MODEL_CHECKPOINT_OUT = Path(WEIGHTS_PILOTNET_FT_HEAD_ONLY_US)

os.makedirs(LOG_DIR, exist_ok=True)

L2_REG_CONST = 1e-4

def create_custom_loss(model, l2_reg_const):
    def custom_loss(y_true, y_pred):
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


def finetune():
    train_loader = DataLoader(str(FINETUNE_DATA_DIR), BATCH_SIZE, is_training=True)
    val_loader   = DataLoader(str(FINETUNE_DATA_DIR), BATCH_SIZE, is_training=False)

    pilotnet = PilotNet(input_shape=(66, 200, 3))
    model = pilotnet.build_model()

    model.load_weights(str(MODEL_CHECKPOINT_IN))

    # Freeze the full feature extractor
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze only the final regression/output layer
    model.layers[-1].trainable = True

    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

    loss_function = create_custom_loss(model, L2_REG_CONST)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_function)

    tensorboard_callback = TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1)
    checkpoint_callback = ModelCheckpoint(
        filepath=str(MODEL_CHECKPOINT_OUT),
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
    )

    history = model.fit(
        train_loader,
        epochs=NUM_EPOCHS_FT,
        validation_data=val_loader,
        callbacks=[tensorboard_callback, checkpoint_callback],
    )

    plot_loss(history)


if __name__ == "__main__":
    finetune()
