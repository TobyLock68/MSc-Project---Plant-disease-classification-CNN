import random
import numpy as np
import cv2

def add_motion_blur(image):
    size = random.randint(5,15)
    kernel = np.zeros((size,size))
    np.fill_diagonal(kernel,1)
    kernel = kernel/size        #normalisation to keep image brightness
    return cv2.filter2D(image, -1, kernel)

def manual_motion_blur(image):
    size = random.randit(7,15)
    kernel = np.eye(size)/size

    blurred = np.zeros_like(image)

    for i in range(3):
        padding = np.pad(image[:,:,i], size//2, mode = 'edge')
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                region = padding[y:y+size, x:x+size]
                blurred[y,x,i] = np.sum(region * kernel)
    
    return blurred.astype(np.unit)

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
    distortions = [add_motion_blur, manual_motion_blur, add_light_intensity, add_noise]

    #random selection
    num_dist = random.randint(1,2)
    select_dist = random.sample(distortions, num_dist)

    #adding selected distortions sequentially
    distorted_img = image.copy()

    for effect in select_dist:
        distorted_img = effect(distorted_img)
    
    return distorted_img
