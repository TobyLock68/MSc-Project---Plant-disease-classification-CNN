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


def add_noise(image):


def apply_layer_1(image):
