
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from data.dataloader_finetune import DataLoader
from models.resnet18 import ResNet18Steering
from config import FINETUNE_DATA_DIR, LOGS_DIR, WEIGHTS_RESNET_US, WEIGHTS_RESNET_PARTIAL_FT_US, BATCH_SIZE, NUM_EPOCHS_FT, LEARNING_RATE


FINETUNE_DATA_DIR = Path(FINETUNE_DATA_DIR)
LOGS_DIR = Path(LOGS_DIR)

MODEL_CHECKPOINT_IN  = Path(WEIGHTS_RESNET_US)
LOG_DIR              = LOGS_DIR    / "logs_partial_backbone_finetuned_ResNet"
MODEL_CHECKPOINT_OUT = Path(WEIGHTS_RESNET_PARTIAL_FT_US)

os.makedirs(LOG_DIR, exist_ok=True)

L2_REG_CONST  = 1e-4


def create_custom_loss(model, l2_reg_const):
    def custom_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        return mse + l2_reg_const * l2_loss
    return custom_loss


def print_model_summary(model):
    trainable_params = 0
    non_trainable_params = 0

    print("\nDetailed Layer Analysis:")
    print("-" * 100)
    print(f"{'Layer #':<8} {'Layer Name':<25} {'Layer Type':<20} {'Trainable':<10} {'Parameters':<15} {'Status'}")
    print("-" * 100)

    for i, layer in enumerate(model.layers):
        layer_params = 0
        trainable_status = layer.trainable

        if trainable_status:
            layer_trainable_params = sum([w.numpy().size for w in layer.trainable_weights])
            layer_non_trainable_params = sum([w.numpy().size for w in layer.non_trainable_weights])
            trainable_params += layer_trainable_params
            non_trainable_params += layer_non_trainable_params
            layer_params = layer_trainable_params + layer_non_trainable_params
        else:
            layer_trainable_params = 0
            layer_non_trainable_params = sum([w.numpy().size for w in layer.weights])
            non_trainable_params += layer_non_trainable_params
            layer_params = layer_non_trainable_params

        print(f"{i:<8} {layer.name:<25} {layer.__class__.__name__:<20} {str(trainable_status):<10} "
              f"{layer_params:<15} {'Frozen' if not trainable_status else 'Trainable'}")

    print("\nParameter Summary:")
    print("-" * 50)
    print(f"Total trainable parameters:     {trainable_params:,}")
    print(f"Total non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters:               {trainable_params + non_trainable_params:,}")

    frozen_layers = sum(1 for layer in model.layers if not layer.trainable)
    trainable_layers = sum(1 for layer in model.layers if layer.trainable)

    print("\nLayer Count Summary:")
    print("-" * 50)
    print(f"Frozen layers:     {frozen_layers}")
    print(f"Trainable layers:  {trainable_layers}")
    print(f"Total layers:      {len(model.layers)}")

    trainable_percent = (trainable_params / (trainable_params + non_trainable_params)) * 100
    print(f"\nTrainable parameters percentage: {trainable_percent:.2f}%")
    print("-" * 50)


def plot_loss(history):
    """Plot the training and validation loss."""
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def freeze_layers(model):
    """
    Partial freeze: freeze first conv block (0-4) and first two ResNet stages (5-18).
    Later ResNet stages and dense layers remain trainable.
    """
    for layer in model.layers[:5]:
        layer.trainable = False

    for layer in model.layers[5:12]:
        layer.trainable = False

    for layer in model.layers[12:19]:
        layer.trainable = False

    return model


def finetune():
    train_loader = DataLoader(str(FINETUNE_DATA_DIR), BATCH_SIZE, is_training=True)
    val_loader   = DataLoader(str(FINETUNE_DATA_DIR), BATCH_SIZE, is_training=False)

    resnet = ResNet18Steering(input_shape=(66, 200, 3))
    model = resnet.build_model()
    print("Loading pre-trained weights from:", MODEL_CHECKPOINT_IN)
    model.load_weights(str(MODEL_CHECKPOINT_IN))

    model = freeze_layers(model)
    print_model_summary(model)

    loss_function = create_custom_loss(model, L2_REG_CONST)
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_function)

    callbacks = [
        TensorBoard(log_dir=str(LOG_DIR), histogram_freq=1, write_graph=True, update_freq='epoch'),
        ModelCheckpoint(
            filepath=str(MODEL_CHECKPOINT_OUT),
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_loader,
        epochs=NUM_EPOCHS_FT,
        validation_data=val_loader,
        callbacks=callbacks,
        verbose=1
    )

    plot_loss(history)

    return model, history


if __name__ == "__main__":
    finetune()
