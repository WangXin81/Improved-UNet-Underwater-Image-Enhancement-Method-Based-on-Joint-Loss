# ori & ori+loss

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def match(x1: torch.Tensor, x2: torch.Tensor):
    # [N, C, H, W]
    diff_y = x2.size()[2] - x1.size()[2]  # H的差值 1
    diff_x = x2.size()[3] - x1.size()[3]  # W的差值 1
    # padding_left, padding_right, padding_top, padding_bottom
    return F.pad(x1, [diff_x, diff_x // 2, diff_y, diff_y // 2])


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),  # ksp:311
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 base_c: int = 64):  # 32
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_conv = DoubleConv(in_channels, base_c, base_c)  # 3 32 32

        self.down1 = nn.MaxPool2d(2, stride=2)
        self._conv1 = DoubleConv(base_c, base_c * 2, base_c * 2)  # 32 64 64
        self.down2 = nn.MaxPool2d(2, stride=2)
        self._conv2 = DoubleConv(base_c * 2, base_c * 4, base_c * 4)  # 64 128 128
        self.down3 = nn.MaxPool2d(2, stride=2)
        self._conv3 = DoubleConv(base_c * 4, base_c * 8, base_c * 8)  # 128 256 256
        self.down4 = nn.MaxPool2d(2, stride=2)
        self.bottom_conv = DoubleConv(base_c * 8, base_c * 8, base_c * 8)  # 256 256 256

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)  # 256 256
        self.conv1_ = DoubleConv(base_c * 8, base_c * 8, base_c * 4)  # 256 256 128
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)  # 128 128
        self.conv2_ = DoubleConv(base_c * 4, base_c * 4, base_c * 2)  # 128 128 64
        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)  # 64 64
        self.conv3_ = DoubleConv(base_c * 2, base_c * 2, base_c)  # 64 64 32
        self.up4 = nn.ConvTranspose2d(base_c, base_c, kernel_size=2, stride=2)  # 32 32
        self.conv4_ = DoubleConv(base_c, base_c, base_c)  # 32 32 32

        self.out_conv = nn.Conv2d(base_c, 3, kernel_size=1)  # 32 3

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)  # 3 128 128 -> 32 128 128

        x = self.down1(x1)  # 32 128 128 -> 32 64 64
        x2 = self._conv1(x)  # 32 64 64 -> 64 64 64
        x = self.down2(x2)  # 64 64 64 -> 64 32 32
        x3 = self._conv2(x)  # 64 32 32 -> 128 32 32
        x = self.down3(x3)  # 128 32 32 -> 128 16 16
        x4 = self._conv3(x)  # 128 16 16 -> 256 16 16
        x = self.down4(x4)  # 256 16 16 -> 256 8 8
        x = self.bottom_conv(x)  # 256 8 8 -> 256 8 8

        x = torch.add(match(self.up1(x), x4), x4)  # 256 8 8 -> 256 16 16
        x = self.conv1_(x)  # 256 16 16 -> 128 16 16
        x = torch.add(match(self.up2(x), x3), x3)  # 128 16 16 -> 128 32 32
        x = self.conv2_(x)  # 128 32 32 -> 64 32 32
        x = torch.add(match(self.up3(x), x2), x2)  # 64 32 32 -> 64 64 64
        x = self.conv3_(x)  # 64 64 64 -> 32 64 64
        x = torch.add(match(self.up4(x), x1), x1)  # 32 64 64 -> 32 128 128
        x = self.conv4_(x)  # 32 128 128 -> 32 128 128

        y = self.out_conv(x)  # 32 128 128 -> 3 128 128

        return {"out": y}


if __name__ == '__main__':
    raw = torch.ones([1, 3, 128, 128])
    model = UNet(3, 32)
    output = model(raw)
    res = output['out']
    print(model)
    print(res.size())
