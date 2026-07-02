import os
import cv2
import multiprocessing
from glob import glob
from pathlib import Path

from coded_parts.distortion_layer_1 import apply_layer_1
from coded_parts.distortion_layer_2 import apply_layer_2

root_dir = Path(__file__).resolve().parent

#actual data directory
input_directory = root_dir / "data" / "plantvillage dataset" / "color"
output_directory = root_dir / "data" / "plantvillage_augmented"

#test directories
#input_directory = str(Path(__file__).parent / "data" / "test_plantvillage")
#output_directory = str(Path(__file__).parent / "data" / "test_plantvillage_augmented")

def distort_single_image(path):
    try:

        img = cv2.imread(path)
        if img is None:
            print(f"--- ERROR READING: {Path} ---")
            return
    
        distorted_img = apply_layer_1(img)
        distorted_img = apply_layer_2(distorted_img)

        current_file_path = Path(path).resolve()
        relative_path = current_file_path.relative_to(input_directory.resolve())
        save_path = output_directory / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), distorted_img)
    
    except Exception as e:
        print(f"--- ERROR PROCESSING FILE {path}: {str(e)} ---")

if __name__ == "__main__":

    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

    image_paths = []

    for ext in extensions:
        image_paths.extend([str(p.resolve()) for p in input_directory.rglob(ext)])

    if len(image_paths) == 0:
        print("NO IMAGES FOUND")
    else:
        #want to run in parallel but not use all cores
        total_cores = multiprocessing.cpu_count()
        num_cores = max(1, total_cores - 2)

        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.map(distort_single_image, image_paths)

    print(f"------- PIPELINE FULLY RUN ------------")