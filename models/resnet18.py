import tensorflow as tf
from tensorflow.keras import layers, Model
 
class ResNet18Steering:
 
    """
    ResNet-18 architecture adapted for steering angle prediction.
    Standard ResNet-18 has [2,2,2,2] basic blocks in each stage.
    """
 
    def __init__(self, input_shape=(66, 200, 3)):
        self.input_shape = input_shape
 
    def basic_block(self, x, filters, stride=1):
 
        """
        Basic ResNet block (no bottleneck)
        """
 
        shortcut = x
 
        # First convolution
 
        x = layers.Conv2D(filters, 3, strides=stride, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
 
        # Second convolution
 
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
 
        # Shortcut connection (identity or projection)
 
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
 
        # Add shortcut
 
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
 
        return x
 
    def build_model(self):
 
        """
 
        Build ResNet-18 model adapted for steering angle prediction
 
        """
 
        inputs = layers.Input(shape=self.input_shape)
 
        # Initial convolution layer
 
        x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
 
        # ResNet stages with [2,2,2,2] configuration
        # Stage 1 (64 channels)
 
        x = self.basic_block(x, 64)
        x = self.basic_block(x, 64)
 
        # Stage 2 (128 channels)
 
        x = self.basic_block(x, 128, stride=2)
        x = self.basic_block(x, 128)
 
        # Stage 3 (256 channels)
 
        x = self.basic_block(x, 256, stride=2)
        x = self.basic_block(x, 256)
 
        # Stage 4 (512 channels)
 
        x = self.basic_block(x, 512, stride=2)
        x = self.basic_block(x, 512)
 
        # Final layers adapted for steering prediction
 
        x = layers.GlobalAveragePooling2D()(x)
 
        # Additional dense layers for steering prediction
 
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='linear')(x)  # Steering angle prediction
 
        model = Model(inputs, outputs, name='resnet18_steering')
 
        return model
 
def create_custom_loss(model, l2_reg_const):
 
    """
 
    Create custom loss function with L2 regularization
 
    """
 
    def custom_loss(y_true, y_pred):
 
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
 
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
 
        return mse + l2_reg_const * l2_loss
 
    return custom_loss
 