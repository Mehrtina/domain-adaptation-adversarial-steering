import tensorflow as tf
from tensorflow.keras import layers, models

class PilotNet(tf.Module):
    """An End-to-End PilotNet to output steering angles given input images of the road ahead"""

    def __init__(self, input_shape=(66,200,3)):
        # Define the model architecture using the Keras Sequential API
        self.model = models.Sequential([
            layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)),
            layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
            layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(1164, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)  # Output layer for steering angle
        ])

    def build_model(self):
        # Return the Keras model
        return self.model

    def predict(self, image):
        # Forward pass to predict steering angle from the image
        raw_output = self.model.predict(image)[0]
        steering_angle = tf.multiply(tf.math.atan(raw_output), 2)  # Apply the atan scaling
        #steering_angle = raw_output
        return steering_angle