from torch import nn


def cal_loss(img1, img2, device='cuda'):
    loss = nn.SmoothL1Loss().to(device)
    return loss(img1, img2)
