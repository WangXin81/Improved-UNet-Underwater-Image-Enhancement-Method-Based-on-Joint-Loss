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


# 残差块
class RB1(nn.Module):
    def __init__(self, in_channels, out_channels, norm='inorm', actv='lrelu'):  # in_channels == out_channels
        super(RB1, self).__init__()
        self.dwconv1 = BasicConv(in_channels, out_channels, groups=out_channels, norm=norm, actv=actv)
        self.dwconv2 = BasicConv(in_channels, out_channels, groups=out_channels, norm=norm, actv=actv)

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x1 + x)
        y = x + x1 + x2
        return y


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


# 注意力模块
class CAM1(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CAM1, self).__init__()
        # c*1*1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # 8 256 8 8
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        y = self.sigmoid(avg_out + max_out)
        return y * x


class CAM2(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CAM2, self).__init__()
        # c*1*1
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.max_pool1 = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.max_pool2 = nn.AdaptiveMaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=2, stride=2, padding=0, bias=False)

        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.max_pool4 = nn.AdaptiveMaxPool2d(4)
        self.conv4 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=4, stride=4, padding=0, bias=False)

        self.conv_final1 = nn.Conv2d(in_channels * 6, in_channels * 6 // ratio, 1, bias=False)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv_final2 = nn.Conv2d(in_channels * 6 // ratio, in_channels, 1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = torch.cat([self.avg_pool1(x), self.max_pool1(x)], dim=1)
        x1 = self.conv1(x1)
        x2 = torch.cat([self.avg_pool2(x), self.max_pool2(x)], dim=1)
        x2 = self.conv2(x2)
        x4 = torch.cat([self.avg_pool4(x), self.max_pool4(x)], dim=1)
        x4 = self.conv4(x4)

        y = torch.cat([x1, x2, x4], dim=1)
        y = self.conv_final2(self.lrelu(self.conv_final1(y)))
        y = self.tanh(y)
        return x * (1. + y)


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        y = self.conv(y)
        # 1*h*w
        y = self.sigmoid(y)
        return y * x


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                   (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / \
                   (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


class AM1(nn.Module):
    def __init__(self, in_channels):
        super(AM1, self).__init__()
        self.cam = CAM1(in_channels)
        self.sam = SAM(in_channels)

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x


class AM2(nn.Module):
    def __init__(self, in_channels):
        super(AM2, self).__init__()
        self.cam = CAM1(in_channels)
        self.sam = SAM(in_channels)

    def forward(self, x):
        xc = self.cam(x)
        xs = self.sam(x)
        return x + xc + xs


# 空间多尺度特征提取模块
class SMFM(nn.Module):
    def __init__(self, in_planes, out_planes, norm='inorm', actv='lrelu', scale=0.1):  # 256 256
        super(SMFM, self).__init__()
        self.scale = scale  # 0.1
        inter_planes = in_planes // 8  # 32

        # 1*1
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, padding=0, norm=norm,
                                  actv='none')  # 256 256, 110
        # 3*3
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=1, padding=0, norm=norm, actv=actv),
            # 256 64, 110
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, norm=norm, actv='none'),
            # 64 64, 311
            CAM1(2 * inter_planes)  # 64 64
        )
        # 5*5
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, norm=norm, actv=actv),  # 256 32, 110
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, norm=norm, actv=actv),
            # 32 64, 311
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=2, dilation=2, norm=norm,
                      actv='none'),
            # 64 64, 312 2
            CAM1(2 * inter_planes)  # 64 64
        )
        # 7*7
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, norm=norm, actv=actv),  # 256 32. 110
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, norm=norm, actv=actv),
            # 32 48, 311
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=1, padding=1, norm=norm,
                      actv=actv),
            # 48 64, 311
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, norm=norm,
                      actv='none'),
            # 64 64, 313 3
            CAM1(2 * inter_planes)  # 64 64
        )
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, padding=0, norm=norm,
                                    actv='none')  # 192 256, 110
        if actv == 'lrelu':
            self.actv = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif actv == 'relu':
            self.actv = nn.ReLU(inplace=True)
        else:
            self.actv = None

    def forward(self, x):  # 256 8 8
        short = self.shortcut(x)  # 256 8 8 -> 256 8 8
        x0 = self.branch0(x)  # 256 8 8 -> 64 8 8 -> 64 8 8
        x1 = self.branch1(x)  # 256 8 8 -> 32 8 8 -> 64 8 8 -> 64 8 8
        x2 = self.branch2(x)  # 256 8 8 -> 32 8 8 -> 48 8 8 -> 64 8 8 -> 64 8 8
        out = torch.cat((x0, x1, x2), 1)  # 192 8 8
        out = self.ConvLinear(out)  # 192 8 8 -> 256 8 8

        out = out * self.scale + short  # 256 8 8
        out = self.actv(out)
        return out


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 base_c: int = 64):  # 32
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.in_conv1 = BasicConv(in_channels, base_c, norm='inorm', actv='lrelu')  # 3 32 311
        self.in_conv2 = RB2(base_c, base_c, norm='inorm', actv='lrelu')  # 32 32

        self.down1 = nn.MaxPool2d(2, stride=2)  # 32 32
        self._adjust1 = BasicConv(base_c, base_c * 2, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 32 64
        self._conv1 = RB2(base_c * 2, base_c * 2, norm='inorm', actv='lrelu')  # 64 64

        self.down2 = nn.MaxPool2d(2, stride=2)  # 64 64
        self._adjust2 = BasicConv(base_c * 2, base_c * 4, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 64 128
        self._conv2 = RB2(base_c * 4, base_c * 4, norm='inorm', actv='lrelu')  # 128 128

        self.down3 = nn.MaxPool2d(2, stride=2)  # 128 128
        self._adjust3 = BasicConv(base_c * 4, base_c * 8, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 128 256
        self._conv3 = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256

        self.down4 = nn.MaxPool2d(2, stride=2)  # 256 256
        self.smfm3 = SMFM(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256
        self._conv4 = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256

        self.up1 = nn.ConvTranspose2d(base_c * 8, base_c * 8, kernel_size=2, stride=2)  # 256 256
        self.conv1_ = RB2(base_c * 8, base_c * 8, norm='inorm', actv='lrelu')  # 256 256
        self.adjust1_ = BasicConv(base_c * 8, base_c * 4, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 256 128

        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 4, kernel_size=2, stride=2)  # 128 128
        self.conv2_ = RB2(base_c * 4, base_c * 4, norm='inorm', actv='lrelu')  # 128 128
        self.adjust2_ = BasicConv(base_c * 4, base_c * 2, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 128 64

        self.up3 = nn.ConvTranspose2d(base_c * 2, base_c * 2, kernel_size=2, stride=2)  # 64 64
        self.conv3_ = RB2(base_c * 2, base_c * 2, norm='inorm', actv='lrelu')  # 64 64
        self.adjust3_ = BasicConv(base_c * 2, base_c, kernel_size=1, stride=1, padding=0, norm='inorm',
                                  actv='lrelu')  # 64 32

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
        x = self.smfm3(x)  # 256 8 8
        x = self._conv4(x)  # 256 8 8 -> 256 8 8

        x5 = self.up1(x)  # 256 8 8 -> 256 16 16
        x = match(x5, x4) + x4  # 256 16 16
        x = self.conv1_(x)  # 256 16 16 -> 256 16 16
        x = x - match(x5, x)  # 256 16 16
        x = self.adjust1_(x)  # 256 16 16 -> 128 16 16

        x6 = self.up2(x)  # 128 16 16 -> 128 32 32
        x = match(x6, x3) + x3  # 128 32 32
        x = self.conv2_(x)  # 128 32 32 -> 128 32 32
        x = x - match(x6, x)  # 128 32 32
        x = self.adjust2_(x)  # 128 32 32 -> 64 32 32

        x7 = self.up3(x)  # 64 32 32 -> 64 64 64
        x = match(x7, x2) + x2  # 64 64 64
        x = self.conv3_(x)  # 64 64 64 -> 64 64 64
        x = x - match(x7, x)  # 64 64 64
        x = self.adjust3_(x)  # 64 64 64 -> 32 64 64

        x8 = self.up4(x)  # 32 64 64 -> 32 128 128
        x = match(x8, x1) + x1  # 32 128 128
        x = self.conv4_(x)  # 32 128 128 -> 32 128 128
        x = x - match(x8, x)  # 32 128 128

        y = self.out_conv(x)  # 32 128 128 -> 3 128 128
        return {"out": y}


if __name__ == '__main__':
    # raw = torch.ones([1, 3, 128, 128])
    # model = UNet(3, 32)
    # output = model(raw)
    # res = output['out']
    # print(model)
    # print(res.size())

    # raw = torch.ones([1, 1, 128, 128])
    # cc = CC(1, 1)
    # output = cc(raw)
    # print(output.size())

    # t1 = torch.ones(1, 3, 3, 3)
    # print(t1)
    # t2 = torch.ones(1, 3, 3, 3)
    # print(t2)
    # print(t1 - t2)

    raw = torch.ones([8, 64, 8, 8])
    cam2 = CAM2(64)
    print(cam2(raw).size())
