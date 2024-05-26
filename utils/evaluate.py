import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import transform
from scipy import ndimage


# 有参考图像增强评价指标
def getPSNR(img1, img2):
    return psnr(img1, img2)


def getSSIM(img1, img2, multichannel):
    return ssim(img1, img2, multichannel=multichannel)  # 求三个通道各自的相似度，再取均值


# 无参考图像增强评价指标
def getUCIQE(img):
    img = (img * 255).astype('uint8')
    img_LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_LAB = np.array(img_LAB, dtype=np.float64)
    coe_Metric = [0.4680, 0.2745, 0.2576]
    img_lum = img_LAB[:, :, 0] / 255.0
    img_a = img_LAB[:, :, 1] / 255.0
    img_b = img_LAB[:, :, 2] / 255.0
    # item-1
    Img_Chr = np.sqrt(np.square(img_a) + np.square(img_b))
    Img_Sat = Img_Chr / np.sqrt(Img_Chr ** 2 + img_lum ** 2)
    Aver_Sat = np.mean(Img_Sat)
    Aver_Chr = np.mean(Img_Chr)
    Var_Chr = np.sqrt(np.mean((np.abs(1 - (Aver_Chr / Img_Chr) ** 2))))
    # item-2
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum) * 0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]
    uciqe = Var_Chr * coe_Metric[0] + con_lum * coe_Metric[1] + Aver_Sat * coe_Metric[2]
    return uciqe


def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    RG = R - G
    YB = (R + G) / 2 - B
    K = R.shape[0] * R.shape[1]
    RG1 = RG.reshape(1, K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanRG = np.sum(RG1) / N
    deltaRG = np.sqrt(np.sum((RG1 - meanRG) ** 2) / N)

    YB1 = YB.reshape(1, K)
    YB1 = np.sort(YB1)
    alphaL = 0.1
    alphaR = 0.1
    YB1 = YB1[0, int(alphaL * K + 1):int(K * (1 - alphaR))]
    N = K * (1 - alphaR - alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB) ** 2) / N)
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaYB ** 2 + deltaRG ** 2)
    return uicm


def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        R = transform.resize(R, (x, y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))
    m = R.shape[0]
    n = R.shape[1]
    k1 = m / patchez
    k2 = n / patchez
    AMEER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = R[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEER = 1 / (k1 * k2) * np.abs(AMEER)
    AMEEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = G[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEG = 1 / (k1 * k2) * np.abs(AMEEG)
    AMEEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = B[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
    AMEEB = 1 / (k1 * k2) * np.abs(AMEEB)
    uiconm = AMEER + AMEEG + AMEEB
    return uiconm


def _uism(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest') + ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest') + ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest') + ndimage.convolve(B, hy, mode='nearest'))
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m % patchez != 0 or n % patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))
    m = SobelR.shape[0]
    n = SobelR.shape[1]
    k1 = m / patchez
    k2 = n / patchez
    EMER = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelR[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max / Min)
    EMER = 2 / (k1 * k2) * np.abs(EMER)

    EMEG = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelG[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max / Min)
    EMEG = 2 / (k1 * k2) * np.abs(EMEG)
    EMEB = 0
    for i in range(0, m, patchez):
        for j in range(0, n, patchez):
            sz = patchez
            im = SobelB[i:i + sz, j:j + sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max / Min)
    EMEB = 2 / (k1 * k2) * np.abs(EMEB)
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism


def getUIQM(x):
    x = x.astype(np.float32)
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm, uicm, uism, uiconm


if __name__ == '__main__':
    img_path = "../results/predict_result/UFO-120/GC/"
    img_names = [i for i in os.listdir(img_path)]
    raw_img_list = [os.path.join(img_path, i) for i in img_names]

    ref_img_path = "../../_datasets/UFO-120/test/ref/"
    ref_img_names = [i for i in os.listdir(ref_img_path)]
    ref_img_list = [os.path.join(ref_img_path, i) for i in ref_img_names]

    sum_psnr = 0.
    sum_ssim = 0.
    sum_uiqm = 0.
    sum_uicm = 0.
    sum_uism = 0.
    sum_uiconm = 0.
    cnt = 0

    for idx in range(0, len(raw_img_list)):
        img = cv2.imread(raw_img_list[idx])
        ref_img = cv2.imread(ref_img_list[idx])

        if img.shape != ref_img.shape:
            print(img_names[idx], "尺寸存在问题。。。")
            continue
        # 计算指标
        cur_psnr = getPSNR(img, ref_img)
        cur_ssim = getSSIM(img, ref_img, multichannel=True)
        cur_uiqm, cur_uicm, cur_uism, cur_uiconm = getUIQM(img)

        sum_psnr += cur_psnr
        sum_ssim += cur_ssim
        sum_uiqm += cur_uiqm
        sum_uicm += cur_uicm
        sum_uism += cur_uism
        sum_uiconm += cur_uiconm
        cnt += 1

        print("有效评价图像个数：", cnt)
        with open(os.path.join("../results/docs/", 'metrics.txt'), 'a') as f:
            f.write("{}:\tPSNR={:.3f}\tSSIM={:.3f}\tUIQM={:.3f}\tUICM={:.3f}\tUISM={:.3f}\tUICONM={:.3f}\n"
                    .format(img_names[idx], cur_psnr, cur_ssim, cur_uiqm, cur_uicm, cur_uism, cur_uiconm))

    mean_psnr = sum_psnr / cnt
    mean_ssim = sum_ssim / cnt
    mean_uiqm = sum_uiqm / cnt
    mean_uicm = sum_uicm / cnt
    mean_uism = sum_uism / cnt
    mean_uiconm = sum_uiconm / cnt

    with open(os.path.join("../results/docs/", 'metrics.txt'), 'a') as f:
        f.write("Average:\tPSNR={:.3f}\tSSIM={:.3f}\tUIQM={:.3f}\tUICM={:.3f}\tUISM={:.3f}\tUICONM={:.3f}\n"
                .format(mean_psnr, mean_ssim, mean_uiqm, mean_uicm, mean_uism, mean_uiconm))
