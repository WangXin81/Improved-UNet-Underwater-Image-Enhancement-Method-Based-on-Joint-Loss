import torch
from torch import nn

import utils.print_utils as utils
from piq import psnr, ssim
from core import l1_loss, l2_loss, perceptual_loss, ssim_loss, tv_loss


# 损失计算
def cal_loss(inputs, target, vgg_module):
    losses = {}
    for name, x in inputs.items():  # out, 8*3*128*128
        # l2 = l2_loss(x, target)
        # loss = l2

        l1 = l1_loss(x, target)
        ssim = ssim_loss(x, target)
        perc = perceptual_loss(vgg_module, nn.L1Loss().to('cuda'), x, target)
        tv = tv_loss(x)
        loss = l1 + 0.1 * ssim + 0.05 * perc + 0.01 * tv

        losses[name] = loss
    if len(losses) == 1:
        return losses['out']
    return losses['out'] + 0.5 * losses['aux']


# 训练一轮
def train_one_epoch(model, optimizer, data_loader, device, epoch, vgg_module, lr_scheduler, print_freq=10, scaler=None):
    model.train()
    # 日志函数
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = cal_loss(output, target, vgg_module)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


# 验证: 指标PSNR SSIM计算 --> 评价分数
def validate(model, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    sum_psnr = 0.
    sum_ssim = 0.

    model.eval()
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 10, header):
            torch.cuda.empty_cache()
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            output[output > 1] = 1
            output[output < 0] = 0

            sum_psnr += psnr(output, target, data_range=1.)
            sum_ssim += ssim(output, target, data_range=1.)

        mean_psnr = sum_psnr / len(data_loader)
        mean_ssim = sum_ssim / len(data_loader)

    return mean_psnr, mean_ssim