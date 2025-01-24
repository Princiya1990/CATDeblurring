# Improving Acne Severity Detection: A GAN Framework with Contour Accentuation for Image Deblurring.
#### Philomina Princiya Mascarenhas, Sannidhan M S, Ancilla J. Pinto, Dabis Camero, Jason Elroy Martis
We present an GAN-powered image processing framework for enhancing and transforming digital sketches. Our approach  leverages Contour Accentuation Techniques (CAT) and Generative Adversarial Networks (GANs) to effectively enhance and deblur sketch contours, enabling accurate transformations into photo-realistic images.

# Process Flow

<img src="images/Sample Output.PNG" alt="Network Architecture" width="600">



# Installation
To install the required libraries, you can use the following command:
```
pip install -r requirements.txt
```
# Module Breakdown
- **Contour Accentuation Technique (CAT):** Enhances image contours through blending and gamma correction techniques.
- **Deblurring Module:** Utilizes GANs to restore blurred facial sketches and improve clarity.
- **Image Translator Module:** Converts sketches into realistic photos while preserving structure and details. 

## Technologies Used
- Python
- TensorFlow
- PyTorch
- GAN-based architectures

## Usage
Clone the repository using 
```bash
git clone https://github.com/Princiya1990/CATDeblurring
```
Verify the folder structure
```
project_root/
├── contouraccentuation/
├── deblurringmodule/
├── imagetranslatormodule/
├── images
```
## Datasets
The datasets for our code are found in  https://shorturl.at/DiN4G

1. **For Running CAT:**
   1. Go to the contouraccentuation module ```cd contouraccentuation```
   2. Place your images in the ```input_images``` folder
   3. Introduce Artifical Blurring by executing the ```python blur.py``` 
   4. run ```python cat.py```
   3. Adjust gamma in the ```adjust_gamma()``` if required function for brightness/contrast fine-tuning.
    
2. **For the Deblurring Module:**
   1. Go to the Deblurring Module ```cd deblurringmodule```
   2. Organize your dataset as follows: (Example folder structure):
```
              datasets/
                blurred/
                   blur_01.jpg
                   blur_02.jpg
                   ...
                clear/
                   clear_01.jpg
                   clear_02.jpg
                    ...
```      
   3. Place the folder of blurred sketches in the ```test``` directory of this project.
   4. Run ```python deblur.py```      
  
3. **To train and run the Image Translator Module**
   1. Go to the Image Translator Module ```cd imagetranslatormodule```
   2. Organize your dataset as follows: (Example folder structure):
```
      data/
              dataset/
                photos/
                   photo_01.jpg
                   photo_02.jpg
                   ...
                sketches/
                   sketch_01.jpg
                   sketch_02.jpg
                    ...
```
   4. Execute ```python sketch_to_image.py```
   5. The results are present in the ```/code/results``` folder.

# Visual Gallery
Here we've shown some sample images of the deblurring process across the adopted datasets, transitioning from blurred inputs to deblurred photos.

<img src="images/gallery.PNG" alt="Network Architecture" width="600">

