import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2


generator_model_path = "deblurring_generator.h5"
generator = load_model(generator_model_path, compile=False)



def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is not None:
        image = cv2.resize(image, target_size) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image, axis=0)  # Add batch dimension
    else:
        print(f"Failed to load image: {image_path}")
        return None


def save_image(image, output_path):
    image = (image * 255).astype(np.uint8)  # Rescale to [0, 255]
    cv2.imwrite(output_path, image)
    print(f"Deblurred image saved: {output_path}")


def deblur_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"deblurred_{filename}")

            # Preprocess the image
            input_image = preprocess_image(input_path)
            if input_image is not None:
                # Generate the deblurred image
                deblurred_image = generator.predict(input_image)[0]
                save_image(deblurred_image, output_path)


input_folder = "blurred_input"  # Folder containing blurred sketches
output_folder = "deblurred_output"  # Folder to save deblurred sketches


deblur_images(input_folder, output_folder)
print("Deblurring completed.")
