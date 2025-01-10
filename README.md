# Improving Acne Severity Detection: A GAN Framework with Contour Accentuation for Image Deblurring.
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
git clone https://github.com/<your-username>/CAT.git](https://github.com/Princiya1990/CATDeblurring
```
1. **For Running CAT:**
   1. Specify the **input_folder** that has the faceimages and the **output_folder** to get the composite sketches.
   2. ```python cat.py```
   3. Adjust gamma in the ```adjust_gamma()``` function for brightness/contrast fine-tuning.
    
2. **For Training the Deblurring Module:**
   1. Place the folder of clear sketches in the root directory of this project. Rename it to ```clear_sketches```
   2. 
   
