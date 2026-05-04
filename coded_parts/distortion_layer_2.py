import albumentations as album
import random
import numpy as np

#image fog function

def add_fog(image):
    transform = album.Compose([
        album.RandomFog(fog_coef_lower = 0.2, fog_coef_upper = 0.4, alpha_coef = 0.4, p = 1.0)
        ])
    foggy_image = transform(image = image)['image']
    return foggy_image
                              



#image lens flare function

def add_lens_flare(image):


#combining layer 2 distortions

def apply_layer_2(image):
    