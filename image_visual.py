import cv2
import numpy as np
import random
from pathlib import Path

import random
import numpy as np
import cv2

def add_motion_blur(image):
    size = random.randint(5,15)
    kernel = np.zeros((size,size))
    np.fill_diagonal(kernel,1)
    kernel = kernel/size        #normalisation to keep image brightness
    return cv2.filter2D(image, -1, kernel)

# def manual_motion_blur(image):
#     size = random.randint(7,15)
#     kernel = np.eye(size)/size

#     blurred = np.zeros_like(image)

#     for i in range(3):
#         padding = np.pad(image[:,:,i], size//2, mode = 'edge')
#         for y in range(image.shape[0]):
#             for x in range(image.shape[1]):
#                 region = padding[y:y+size, x:x+size]
#                 blurred[y,x,i] = np.sum(region * kernel)
    
#     return blurred.astype(np.uint8)

def add_light_intensity(image):
    #HLS section
    brightness = random.uniform(0.5,1.5)    #dictates whether image gets lighter or darker

    #switches RGB to HLS for easier brightness alteration
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32) 
    hls[:,:,1] = hls[:,:,1]*brightness

    #returns back to RGB from HLS
    transition_img = cv2.cvtColor(np.clip(hls, 0, 255).astype(np.uint8),cv2.COLOR_HLS2BGR)

    #Linear blending

    #initialise hyperparameters:
    alpha = random.uniform(0.9 , 1.1)
    beta = random.uniform(-10, 10)

    #coverting pixels to floats to avoid errors around 0 - 255
    img_float = transition_img.astype(np.float32)
    img_mean = np.mean(img_float)

    blended_img = alpha*img_float + (1-alpha)*img_mean + beta

    return np.clip(blended_img, 0, 255).astype(np.uint8)


def add_noise(image):

    img_float = image.astype(np.float32)

    #Logic for random assignment of gaussian or stochastic noise
    mode = random.randint(0,1)

    if mode == 1: #stochastic
        noise = np.random.randint(-30,30, image.shape)
    else:   #gaussian
        hyper_p = random.randint(10,50)
        noise = np.random.normal(0, hyper_p, image.shape)
    
    noisy_img = img_float + noise

    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_layer_1(image):
    distortions = [add_motion_blur, add_light_intensity, add_noise]

    #random selection
    num_dist = random.randint(1,2)
    select_dist = random.sample(distortions, num_dist)

    #adding selected distortions sequentially
    distorted_img = image.copy()

    for effect in select_dist:
        distorted_img = effect(distorted_img)
    
    return distorted_img


import albumentations as album
import random
import numpy as np

#image fog function

def add_fog(image):
    transform = album.Compose([
        album.RandomFog(fog_coef_range=(0.2, 0.4), alpha_coef = 0.4, p = 1.0)
        ])
    foggy_image = transform(image = image)['image']
    return foggy_image
                              

#image lens flare function

def add_lens_flare(image):
    transform = album.Compose([
        album.RandomSunFlare(flare_roi=[0,0,1,0.7], src_radius = 400, src_color = [255,245,215],
                             angle_range = [0,1], num_flare_circles_range=[9,10], method = "overlay")
    ])
    lens_flare = transform(image = image)['image']
    return lens_flare

#maybe change method to physics_based

#combining layer 2 distortions

def apply_layer_2(image):

#may add new logic to decrease this layers occurence

    distortions = [None, add_lens_flare, add_fog]
    
    choice = random.choice(distortions)

    if choice is None:
        return image
    
    return choice(image)


# 2. Setup your local paths
input_image_path = Path("/Users/tobylock/Desktop/MSc-Project---Plant-disease-classification-CNN/data/plantvillage dataset/color/Apple___Apple_scab/0cbfa4fa-63d8-43ce-9385-ff140e524b69___FREC_Scab 3164.JPG")
output_dir = Path("my_custom_distortions")
output_dir.mkdir(exist_ok=True)

# 3. Load the original image using OpenCV
img = cv2.imread(str(input_image_path))

if img is None:
    raise FileNotFoundError(f"Could not load your image at: {input_image_path}. Check the path string!")

# Save a copy of the pristine baseline image
cv2.imwrite(str(output_dir / "0_original.jpg"), img)
print("Saved: 0_original.jpg")

# 4. Process Layer 1 Distortions (OpenCV / NumPy)
print("\n--- Processing Layer 1 (Sensor/Camera Environment) ---")

blur_sample = add_motion_blur(img.copy())
cv2.imwrite(str(output_dir / "layer1_motion_blur.jpg"), blur_sample)
print("Saved: layer1_motion_blur.jpg")

light_sample = add_light_intensity(img.copy())
cv2.imwrite(str(output_dir / "layer1_light_intensity.jpg"), light_sample)
print("Saved: layer1_light_intensity.jpg")

noise_sample = add_noise(img.copy())
cv2.imwrite(str(output_dir / "layer1_noise.jpg"), noise_sample)
print("Saved: layer1_noise.jpg")

# 5. Process Layer 2 Distortions (Albumentations Weather Effects)
print("\n--- Processing Layer 2 (Atmospheric/Weather Effects) ---")

fog_sample = add_fog(img.copy())
cv2.imwrite(str(output_dir / "layer2_fog.jpg"), fog_sample)
print("Saved: layer2_fog.jpg")

flare_sample = add_lens_flare(img.copy())
cv2.imwrite(str(output_dir / "layer2_lens_flare.jpg"), flare_sample)
print("Saved: layer2_lens_flare.jpg")

print(f"\nAll set! Open the '{output_dir}/' directory to view your thesis figures.")