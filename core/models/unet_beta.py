# ori+loss+mrm

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


def match(x1: torch.Tensor, x2: torch.Tensor):  # 主要针对测试时不同分辨率的图片，让小的填充成大的
    # [N, C, H, W]
    diff_y = x2.size()[2] - x1.size()[2]  # H的差值 1
    diff_x = x2.size()[3] - x1.size()[3]  # W的差值 1
    # padding_left, padding_right, padding_top, padding_bottom
    return F.pad(x1, [diff_x, diff_x // 2, diff_y, diff_y // 2])


# 基本卷积层
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                 norm='inorm', actv='lrelu'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias, dilation=dilation, groups=groups)
        if norm == 'inorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'bnorm':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        if actv == 'lrelu':
            self.actv = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif actv == 'relu':
            self.actv = nn.ReLU(inplace=True)
        else:
            self.actv = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.actv is not None:
            x = self.actv(x)
        return x


# 双卷积 --> 残差块
class RB2(nn.Module):
    def __init__(self, in_channels, out_channels, norm='inorm', actv='lrelu'):  # 32 64
        super(RB2, self).__init__()
        growth_rate = in_channels // 2
        self.c1 = BasicConv(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, norm=norm, actv=actv)
        self.c2 = BasicConv(growth_rate, growth_rate * 2, kernel_size=3, stride=1, padding=1, norm=norm, actv=actv)
        self.c3 = BasicConv(growth_rate * 2, growth_rate * 3, kernel_size=3, stride=1, padding=1, norm=norm, actv=actv)
        self.ac1 = BasicConv(growth_rate * 6, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, actv=actv)
        self.ac2 = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, norm=norm, actv=actv)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        y1 = self.ac1(torch.concat([x1, x2, x3], dim=1))
        y2 = self.ac2(x)
        return y1 + y2


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 base_c: int = 64):  # 32
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_conv1 = BasicConv(in_channels, base_c, norm='inorm', actv='lrelu')  # 3 32 311
        self.in_conv2 = RB2(base_c, base_c, norm='inorm', actv='lrelu')  # 32 32

        self.down1 = nn.MaxPool2d(2, stride=2)  # 32 32
        self._adjust1 = BasicConv(base_c, base_c * 2, norm='inorm', actv='lrelu')  # 32 64
        self._conv1 = RB2(base_c * 2, base_c * 2, norm='inorm', actv='lrelu')  # 64 64
        self.down2 = nn.MaxPool2d(2, stride=2)  # 64 64
        self._adjust2 = BasicConv(base_c * 2, base_c * 4, norm='inorm', actv='lrelu')  # 64 128
        self._conv2 = RB2(base_c * 4, base_c * 4, norm='inorm', actv='lrelu')  # 128 128
        self.down3 = nn.MaxPool2d(2, stride=2)  # 128 128
        self._adjust3 = BasicConv(base_c * 4, base_c * 8, norm='inorm', actv='lrelu')  # 128 256
        self._conv3 = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256
        self.down4 = nn.MaxPool2d(2, stride=2)  # 256 256
        self._conv4 = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)  # 256 256
        self.conv1_ = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256
        self.adjust1_ = BasicConv(base_c * 8, base_c * 4, norm='inorm', actv='lrelu')  # 256 128
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)  # 128 128
        self.conv2_ = RB2(base_c * 4, base_c * 4, norm='inorm', actv='lrelu')  # 128 128
        self.adjust2_ = BasicConv(base_c * 4, base_c * 2, norm='inorm', actv='lrelu')  # 128 64
        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)  # 64 64
        self.conv3_ = RB2(base_c * 2, base_c * 2, norm='inorm', actv='lrelu')  # 64 64
        self.adjust3_ = BasicConv(base_c * 2, base_c, norm='inorm', actv='lrelu')  # 64 32
        self.up4 = nn.ConvTranspose2d(base_c, base_c, kernel_size=2, stride=2)  # 32 32
        self.conv4_ = RB2(base_c, base_c, norm='inorm', actv='lrelu')  # 32 32

        self.out_conv = nn.Conv2d(base_c, 3, kernel_size=3, stride=1, padding=1)  # 32 3 311

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.in_conv1(x)  # 3 128 128 -> 32 128 128
        x1 = self.in_conv2(x)  # 32 128 128 -> 32 128 128

        x = self.down1(x1)  # 32 128 128 -> 32 64 64
        x = self._adjust1(x)  # 32 64 64 -> 64 64 64
        x2 = self._conv1(x)  # 64 64 64 -> 64 64 64

        x = self.down2(x2)  # 64 64 64 -> 64 32 32
        x = self._adjust2(x)  # 64 32 32 -> 128 32 32
        x3 = self._conv2(x)  # 128 32 32 -> 128 32 32

        x = self.down3(x3)  # 128 32 32 -> 128 16 16
        x = self._adjust3(x)  # 128 16 16 -> 256 16 16
        x4 = self._conv3(x)  # 256 16 16 -> 256 16 16

        x = self.down4(x4)  # 256 16 16 -> 256 8 8
        x = self._conv4(x)  # 256 8 8 -> 256 8 8

        x = self.up1(x)  # 256 8 8 -> 256 16 16
        x = match(x, x4) + x4  # 256 16 16
        x = self.conv1_(x)  # 256 16 16 -> 256 16 16
        x = self.adjust1_(x)  # 256 16 16 -> 128 16 16

        x = self.up2(x)  # 128 16 16 -> 128 32 32
        x = match(x, x3) + x3  # 128 32 32
        x = self.conv2_(x)  # 128 32 32 -> 128 32 32
        x = self.adjust2_(x)  # 128 32 32 -> 64 32 32

        x = self.up3(x)  # 64 32 32 -> 64 64 64
        x = match(x, x2) + x2  # 64 64 64
        x = self.conv3_(x)  # 64 64 64 -> 64 64 64
        x = self.adjust3_(x)  # 64 64 64 -> 32 64 64

        x = self.up4(x)  # 32 64 64 -> 32 128 128
        x = match(x, x1) + x1  # 32 128 128
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

    # raw = torch.ones([1, 1, 128, 128])
    # cc = CC(1, 1)
    # output = cc(raw)
    # print(output.size())

    # t1 = torch.ones(1, 3, 3, 3)
    # print(t1)
    # t2 = torch.ones(1, 3, 3, 3)
    # print(t2)
    # print(t1 - t2)

