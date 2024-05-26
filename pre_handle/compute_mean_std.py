import os
from PIL import Image
import numpy as np


def main():
    img_channels = 3
    img_dir = "../../_datasets/val/"
    assert os.path.exists(img_dir), f"image dir: '{img_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir)]
    cumulative_mean = np.zeros(img_channels)
    cumulative_std = np.zeros(img_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        img = np.array(Image.open(img_path)) / 255.

        cumulative_mean += img.mean(axis=(0, 1))
        cumulative_std += img.std(axis=(0, 1))

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    print(f"mean: {mean}")
    print(f"std: {std}")


if __name__ == '__main__':
    main()
