import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import timedelta
from time import time

"""
    计算图像的mean和std
"""


def get_relative_paths(root_dir):
    relative_paths = []
    for root, directories, files in os.walk(root_dir):
        for file in files:
            if file[0] == ".":
                continue
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, root_dir)
            relative_paths.append(os.path.join(root_dir, relative_path))
    return relative_paths


def main(path: str, mode: int = 1):
    """
        计算图像的mean和std
    :param path:
    :param mode: 0灰度，1RGB，-1透明通道
    :return:
    """
    start = time()

    img_name_list = get_relative_paths(path)
    cumulative_mean = 0
    cumulative_std = 0
    if mode == 0:
        axis = None
    elif mode == 1:
        axis = (0, 1)
    else:
        raise "mode error"
    for img_name in tqdm(img_name_list):
        img = cv2.imread(img_name, mode)
        mean = np.mean(img, axis)
        std = np.std(img, axis)
        cumulative_mean += mean
        cumulative_std += std

    mean = cumulative_mean / len(img_name_list) / 256
    std = cumulative_std / len(img_name_list) / 256
    print(f"mean: {np.array2string(np.round(mean, 5), separator=', ')}")
    print(f"std: {np.array2string(np.round(std, 5), separator=', ')}")
    print(f"用时: {timedelta(seconds=int(time() - start))}")


if __name__ == '__main__':
    main("data/brain1/prostate/mix")
    """
        brain1
        mean: [0.43879, 0.18755, 0.24402]
std: [0.09237, 0.18096, 0.04245]
         vali
 mean: [0.4337, 0.1738, 0.2465]
std: [0.09317, 0.17729, 0.0395 ]        
        fruit1
            mean: [0.44115, 0.56709, 0.63928]
            std: [0.2581 , 0.23213, 0.21401]
            用时: 0:00:00
        che2
            mean: [0.47104, 0.47104, 0.47104]
            std: [0.22306, 0.22306, 0.22306]
            用时: 0:00:53
        beaast1
            mean: [0.74264, 0.55969, 0.74975]
            std: [0.0905 , 0.16546, 0.12208]
            用时: 0:00:04
        breast2
            mean: [0.73976, 0.59303, 0.75426]
            std: [0.08841, 0.15652, 0.11861]
            用时: 0:00:25
    """
"""
prosrate 
"""