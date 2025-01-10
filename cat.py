import os
import numpy as np
import cv2

#functions
def render(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0, 0)  # Gaussian Blur for smoothening
    img_blend = cv2.divide(img_gray, img_blur, scale=256)  # Blend grayscale with blurred version
    return img_blend

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
input_folder = "input_images"  # Replace with your input folder path
output_folder = "output_images"  # Replace with your output folder path
os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):  
        input_image_path = os.path.join(input_folder, filename)
        output_image_path = os.path.join(output_folder, f"processed_{filename}")
        image_array = cv2.imread(input_image_path)
        if image_array is not None:  
            # Apply the CAT process
            img_blend = render(image_array)
            img_gamma = adjust_gamma(img_blend, gamma=.001)
            cv2.imwrite(output_image_path, img_gamma)
            print(f"Processed image saved: {output_image_path}")
        else:
            print(f"Failed to process image: {input_image_path}")


print("Batch processing complete!")
