import os
import time
import datetime
import argparse
import torch
from torchvision.models import vgg19

from utils import train_and_val
from core import UNet
from core import UIDataset
from core import transforms as T
from core import create_lr_scheduler


# 训练集预处理
class PresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        min_size = int(0.5 * base_size)  # 256*0.5=128  512*0.5=256
        max_size = int(1.2 * base_size)  # 256*1.2=307  512*1.2=614
        # 随机缩放
        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            # 水平翻转
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            # 上下翻转
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([
            # 随机裁剪
            T.RandomCrop(crop_size),
            # 转换为张量并归一化
            T.ToTensor(),
            # 标准化
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


# 验证集预处理
class PresetVal:
    def __init__(self, mean, std):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transform(train, mean, std):
    base_size = 256
    crop_size = 128

    if train:
        return PresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return PresetVal(mean=mean, std=std)


def create_model(in_channels, base_c):
    model = UNet(in_channels=in_channels, base_c=base_c)
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_size = args.batch_size
    print("batch_size = ", batch_size)
    base_c = args.base_c
    print("base_c = ", base_c)

    # # UIEB训练集原图标准化数据
    # train_mean = (0.271, 0.496, 0.498)
    # train_std = (0.134, 0.159, 0.162)
    # # UIEB测试集原图标准化数据
    # val_mean = (0.259, 0.475, 0.488)
    # val_std = (0.141, 0.166, 0.172)

    # UFO-120训练集原图标准化数据
    train_mean = (0.218, 0.468, 0.460)
    train_std = (0.159, 0.187, 0.187)
    # UFO-120测试集原图标准化数据
    val_mean = (0.245, 0.473, 0.463)
    val_std = (0.177, 0.180, 0.181)

    # 用来记录训练过程信息
    results_file = "./results/docs/{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_dataset = UIDataset(args.data_path,
                              train=True,
                              transforms=get_transform(train=True, mean=train_mean, std=train_std))
    val_dataset = UIDataset(args.data_path,
                            train=False,
                            transforms=get_transform(train=False, mean=val_mean, std=val_std))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True)

    # 定义模型并投放到GPU
    model = create_model(in_channels=3, base_c=base_c)
    model.to(device)
    print("model: ", model)

    # 感知损失所用vgg模型
    vgg = vgg19(pretrained=True, progress=True).features
    vgg.eval()
    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False
    vgg_module = vgg[0:36]
    vgg_module.to(device)
    print('==========' * 10)
    # print("vgg_module ", vgg_module)

    # 定义优化器并初始化模型参数
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 定义学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 断点续训操作
    if args.resume:
        print("继续训练。。。")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 记录最好的验证集分数
    best_val_score = 0

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练，返回损失和当前学习率
        mean_loss, lr = train_and_val.train_one_epoch(model, optimizer, train_loader, device, epoch, vgg_module,
                                                      lr_scheduler=lr_scheduler, print_freq=args.print_freq,
                                                      scaler=scaler)
        # 验证，返回评价指标
        mean_psnr, mean_ssim = train_and_val.validate(model, val_loader, device=device)
        # 验证分数，以此对比模型好坏
        val_score = mean_psnr + mean_ssim

        # 保存模型
        save_last = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_last["scaler"] = scaler.state_dict()
        torch.save(save_last, "results/weights/last.pth")
        if args.save_best is True:
            if val_score >= best_val_score:
                best_val_score = val_score
                save_best = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch,
                             "args": args}
                if args.amp:
                    save_best["scaler"] = scaler.state_dict()
                torch.save(save_best, "results/weights/best.pth")

        # 记录训练过程
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证分数
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            val_info = f"mean_psnr: {mean_psnr:.4f}\n" \
                       f"mean_ssim: {mean_ssim:.4f}\n" \
                       f"val_score: {val_score:.4f}\n" \
                       f"current best_val_score: {best_val_score:.4f}\n"
            f.write(train_info + val_info + "\n")
        print(train_info + val_info)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training")
    # 数据集路径
    parser.add_argument("--data-path", default="../_datasets/UFO-300/", help="data root")
    # 基础通道数
    parser.add_argument("--base-c", default=32, type=int)
    # 选用设备
    parser.add_argument("--device", default="cuda", help="training device")
    # 批大小
    parser.add_argument("--batch-size", default=8, type=int)
    # 训练轮次
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to train")
    # 初始学习率
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    # 学习率动量
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    # 学习率的权重衰减系数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 打印频率,单位：批次/迭代次数
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # 断点续训
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 开始轮次
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='start epoch')
    # 保存最好的模型权重
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters 混合精度计算
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists("results/weights"):
        os.mkdir("results/weights")
    main(args)
