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
    distortions = [None, add_lens_flare, add_fog]
    
    choice = random.choice(distortions)

    if choice is None:
        return image
    
    return choice(image)