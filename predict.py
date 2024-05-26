import os
import time
import torch
from torchvision import transforms
from PIL import Image
from core import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "results/weights/UIEB/ori+full/best.pth"
    img_path = "../_datasets/val/"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."

    # # UIEB测试集原图标准化数据
    # mean = (0.259, 0.475, 0.488)
    # std = (0.141, 0.166, 0.172)

    # UFO-120测试集原图标准化数据
    # mean = (0.245, 0.473, 0.463)
    # std = (0.177, 0.180, 0.181)

    # LFITW测试集原图标准化数据
    mean = (0.230, 0.400, 0.349)
    std = (0.092, 0.112, 0.096)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model  注意模型的选择
    model = UNet(in_channels=3, base_c=32)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # load image
    img_names = [i for i in os.listdir(img_path)]
    raw_img_list = [os.path.join(img_path, i) for i in img_names]
    for idx in range(0, len(raw_img_list)):
        original_img = Image.open(raw_img_list[idx]).convert('RGB')

        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension  升维
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init model
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            output = model(img.to(device))
            t_end = time_synchronized()
            print("inference time: {}".format(t_end - t_start))

            prediction = output['out']
            prediction = torch.squeeze(prediction)
            prediction = prediction.to("cpu")
            prediction = prediction.numpy().transpose((1, 2, 0))
            prediction[prediction > 1] = 1
            prediction[prediction < 0] = 0
            prediction = prediction * 255
            prediction = prediction.astype("uint8")
            res = Image.fromarray(prediction, mode='RGB')
            res.save(os.path.join("./results/predict_result/output/", img_names[idx]))
            print(img_names[idx] + " 处理完毕！完成个数：" + str(idx + 1))


if __name__ == '__main__':
    main()
