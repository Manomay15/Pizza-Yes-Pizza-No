import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):
    model = keras.models.load_model(weights_file)
    data = np.ndarray(shape=(1, 255, 255, 3), dtype=np.float32)
    image = img
    size = (255, 255)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction