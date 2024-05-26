import torch
from torch import nn


# ======================总变分损失函数部分====================
# 可有效去除噪声
def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def tv_loss(x):
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)


class TV_Loss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TV_Loss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.shape[0]
        return self.TVLoss_weight * tv_loss(x) / batch_size


def cal_loss(img1, device='cuda'):
    creation = TV_Loss().to(device)
    loss = creation(img1)
    return loss


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(size=(1, 3, 256, 256), dtype=torch.float32, device=device)
    print(x)
    loss = cal_loss(x)
    print(loss)  # tensor(0.6671, device='cuda:0')
