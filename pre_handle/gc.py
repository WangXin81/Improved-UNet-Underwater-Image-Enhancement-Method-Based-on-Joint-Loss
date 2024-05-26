import os
import numpy as np
import cv2
from PIL import Image


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


data_root = "./test/"
img_names = [i for i in os.listdir(os.path.join(data_root, "raw"))]
raw_img_list = [os.path.join(data_root, "raw", i) for i in img_names]

for idx in range(0, len(raw_img_list)):

    img_gray = cv2.imread(raw_img_list[idx], 0)  # 灰度图读取，用于计算gamma值
    img = cv2.imread(raw_img_list[idx])  # 原图读取
    mean = np.mean(img_gray)
    # gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
    gamma_val = 0.7
    image_gamma_correct = gamma_trans(img, gamma_val)  # gamma变换

    rgb_img = Image.fromarray(image_gamma_correct)
    rgb_img.save(os.path.join(data_root, "gc", img_names[idx]))
    print(img_names[idx] + " 处理完毕！完成个数：" + str(idx + 1))