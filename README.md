# Improving Acne Severity Detection: A GAN Framework with Contour Accentuation for Image Deblurring. (Authors Add)
We present an GAN-powered image processing framework for enhancing and transforming digital sketches. Our approach  leverages Contour Accentuation Techniques (CAT) and Generative Adversarial Networks (GANs) to effectively enhance and deblur sketch contours, enabling accurate transformations into photo-realistic images.

# Sample Output

<img src="images/Sample Output.PNG" alt="Network Architecture" width="600">

# Module Breakdown
- **Contour Accentuation Technique (CAT):** Enhances image contours through blending and gamma correction techniques.
- **Deblurring Module:** Utilizes GANs to restore blurred facial sketches and improve clarity.
- **Image Translator Module:** Converts sketches into realistic photos while preserving structure and details. 

# Technologies Used
- Python
- TensorFlow
- PyTorch
- GAN-based architectures

# Usage
Clone the repository using 
```bash
git clone https://github.com/Princiya1990/CATDeblurring
```
1. **For Running CAT:**
   1. Specify the **input_folder** that has the faceimages and the **output_folder** to get the composite sketches.
   2. ```python cat.py```
   3. Adjust gamma in the ```adjust_gamma()``` function for brightness/contrast fine-tuning.
    
2. **For Training the Deblurring Module:**
   1. Place the folder of clear sketches in the root directory of this project. Rename it to ```clear_sketches```
   2. Place the folder of blurred sketches in the root directory of this project. Rename it to ```blurred_sketches```
   3. Run ```python deblurring.py```
   
3. **To Infer and Run the Deblurring Module:**
   1. Place your Blurred sketches in the folder named **blurred_input**. We prefer (.jpg, .png ) as extensions.
   2. Run ```python deblur_inference.py```
   3. Deblurred images are saved in a folder named **deblurred_output**.
   4. The deblurred filenames will be prefixed with **deblurred_**.
  
4. **To train and run the Image Translator Module**
   1. For training this module place the skecthes are image pair into their corresponding folder named ```sketches``` and ```images```
   2. Run ```python sketch_to_image.py```
   3. After training, the generated images in the ```generated_images/``` folder.
