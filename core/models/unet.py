from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # spk：1 1 3
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),  # 2 0 2
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # 2 0 2
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]  # H的差值 1
        diff_x = x2.size()[3] - x1.size()[3]  # W的差值 1
        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x, diff_x // 2, diff_y, diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 1 0 1
        )


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 base_c: int = 64):  # 32
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_conv = DoubleConv(in_channels, base_c)  # 3 32

        self.down1 = Down(base_c, base_c * 2)  # 32 64
        self.down2 = Down(base_c * 2, base_c * 4)  # 64 128
        self.down3 = Down(base_c * 4, base_c * 8)  # 128 256
        self.down4 = Down(base_c * 8, base_c * 16)  # 256 512

        self.bottom_conv = DoubleConv(base_c * 16, base_c * 16)  # 512 512

        self.up1 = Up(base_c * 16, base_c * 8)  # 512 256
        self.up2 = Up(base_c * 8, base_c * 4)  # 256 128
        self.up3 = Up(base_c * 4, base_c * 2)  # 128 64
        self.up4 = Up(base_c * 2, base_c)  # 64 32

        self.out_conv = OutConv(base_c, 3)  # 32 3

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)  # 3 120 120 -> 32 120 120
        x2 = self.down1(x1)  # 32 120 120 -> 64 60 60
        x3 = self.down2(x2)  # 64 60 60 -> 128 30 30
        x4 = self.down3(x3)  # 128 30 30 -> 256 15 15
        x5 = self.down4(x4)  # 256 15 15 -> 512 7 7

        x = self.bottom_conv(x5)  # 512 7 7 -> 512 7 7

        x = self.up1(x, x4)  # 512 7 7 -> 256 15 15
        x = self.up2(x, x3)  # 256 15 15 -> 128 30 30
        x = self.up3(x, x2)  # 128 30 30 -> 64 60 60
        x = self.up4(x, x1)  # 64 60 60 -> 32 120 120
        y = self.out_conv(x)  # 32 120 120 -> 3 120 120
        return {"out": y}


if __name__ == '__main__':
    raw = torch.ones([1, 3, 120, 120])
    model = UNet(3, 32)
    output = model(raw)
    res = output['out']
    print(res.size())
