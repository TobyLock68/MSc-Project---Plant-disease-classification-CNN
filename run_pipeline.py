import os
import cv2
import multiprocessing
from glob import glob

from coded_parts.distortion_layer_1 import apply_layer_1
from coded_parts.distortion_layer_2 import apply_layer_2

input_directory = "data/plantvillage dataset"
output_directory = "data/plantvillage_augmented"

def distort_single_image(path):
    img = cv2.imread(path)
    if img is None:
        return
    
    distorted_img = apply_layer_1(img)
    distorted_img = apply_layer_2(distorted_img)

    save_path = path.replace(input_directory, output_directory)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, distorted_img)


if __name__ == "__main__":
    image_paths = glob(f"{input_directory}/**/*.[jJ][pP]*[gG]", recursive=True) + glob(f"{input_directory}/**/*.png", recursive=True)

    #want to run in parallel but not use all cores

    total_cores = multiprocessing.cpu_count()
    num_cores = max(1, total_cores - 2)

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.map(distort_single_image, image_paths)
