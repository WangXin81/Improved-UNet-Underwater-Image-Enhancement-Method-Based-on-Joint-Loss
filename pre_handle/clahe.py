from PIL import Image
import numpy as np
import cv2
import os

data_root = "../../_datasets/UIEB/test/"
img_names = [i for i in os.listdir(os.path.join(data_root, "raw")) if i.endswith(".png")]
raw_img_list = [os.path.join(data_root, "raw", i) for i in img_names]

for idx in range(0, len(raw_img_list)):
    img = Image.open(raw_img_list[idx]).convert('RGB')
    img = np.uint8(img)

    imgr = img[:, :, 0]
    imgg = img[:, :, 1]
    imgb = img[:, :, 2]

    claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 18))
    claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 18))
    claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(10, 18))
    cllr = claher.apply(imgr)
    cllg = claheg.apply(imgg)
    cllb = claheb.apply(imgb)

    rgb_img = Image.fromarray(np.dstack((cllr, cllg, cllb)))
    rgb_img.save(os.path.join(data_root, "clahe", img_names[idx]))
    print(img_names[idx] + " 处理完毕！完成个数：" + str(idx+1))
