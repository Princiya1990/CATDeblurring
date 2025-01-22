import os
from PIL import Image, ImageFilter

# Directories for input and output images
input_dir = 'input_images'  # Replace with your input folder path
output_dir = 'blurred_images'  # Replace with your output folder path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpeg', '.jpg', '.png')):  # Only process image files
        input_path = os.path.join(input_dir, filename)
        
        # Open the image
        with Image.open(input_path) as img:
            # Apply Gaussian Blur with radius=5
            blurred_img = img.filter(ImageFilter.GaussianBlur(5))
            
            # Save the blurred image to the output folder
            output_path = os.path.join(output_dir, filename)
            blurred_img.save(output_path)

print(f"All images have been processed and saved in the '{output_dir}' folder.")
