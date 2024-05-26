import torch
from torch import nn


def cal_loss(img1, img2, device='cuda'):
    loss = nn.L1Loss().to(device)
    return loss(img1, img2)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 256, 256), dtype=torch.float32, device=device)
    y = torch.randn((1, 3, 256, 256), dtype=torch.float32, device=device)
    print(x)
    print(y)
    loss = cal_loss(x, y)
    print(loss)  # tensor(1.1295, device='cuda:0')
