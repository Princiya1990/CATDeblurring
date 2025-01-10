import os
import numpy as np
import cv2

# Function to apply the blending step
def render(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)  # Gaussian Blur for smoothening
    img_blend = cv2.divide(img_gray, img_blur, scale=256)  # Blend grayscale with blurred version
    return img_blend

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Paths for the input and output folders
input_folder = "input_images"  # Replace with your input folder path
output_folder = "output_images"  # Replace with your output folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  # Check for valid image extensions
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f"processed_{filename}")

        # Read the input image
        image_array = cv2.imread(input_image_path)

        if image_array is not None:  # Check if the image was read correctly
            # Apply the CAT process
            img_blend = render(image_array)
            img_gamma = adjust_gamma(img_blend, gamma=.001)  # Adjust gamma for better contrast

            # Save the processed image
            cv2.imwrite(output_image_path, img_gamma)
            print(f"Processed image saved: {output_image_path}")
        else:
            print(f"Failed to process image: {input_image_path}")

print("Batch processing complete!")
