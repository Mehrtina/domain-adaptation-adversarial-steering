import cv2
import numpy as np


def preprocess_image(img_path, target_size=(200, 66)):
    """Load, crop, resize, and normalise an image to YUV format.

    Crops the bottom 800 pixels, resizes to (width=200, height=66),
    converts BGR to YUV, and normalises pixel values to [0, 1].
    Used for Australian evaluation and fine-tuning data.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from path: {img_path}")

    img = img[-800:, :, :]  

    img_resized = cv2.resize(img, target_size)

    img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2YUV)
    img_yuv_normalized = img_yuv / 255.0

    return img_yuv_normalized
