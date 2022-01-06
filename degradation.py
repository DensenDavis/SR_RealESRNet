import torch
import cv2
import numpy as np
import random
import math
from torch.nn import functional as F
from torch.nn.functional import conv2d
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_mixed_kernels, circular_lowpass_kernel
from config import Configuration
cfg = Configuration()


class Degradation:
    def __init__(self):
        # self.usm_sharpener = USMSharp().cuda()
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        return
    
    def apply_degradations(self,data_batch):
        gt_batch = data_batch["gt_img"].to(cfg.device)
        k1 = data_batch["k1"].to(cfg.device)
        k2 = data_batch["k2"].to(cfg.device)
        sinc_k = data_batch["sinc_k"].to(cfg.device)

        # gt_usm = self.usm_sharpener(gt_batch)
        ori_h, ori_w = gt_batch.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt_batch, k1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], cfg.resize_prob)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, cfg.resize_range[1])
        elif updown_type == 'down':
            scale = np.random.uniform(cfg.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # noise
        gray_noise_prob = cfg.gray_noise_prob
        if np.random.uniform() < cfg.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=cfg.noise_range, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=cfg.poisson_scale_range,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < cfg.second_blur_prob:
            out = filter2D(out, k2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], cfg.resize_prob2)[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, cfg.resize_range2[1])
        elif updown_type == 'down':
            scale = np.random.uniform(cfg.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / cfg.scale * scale), int(ori_w / cfg.scale * scale)), mode=mode)
        # noise
        gray_noise_prob = cfg.gray_noise_prob2
        if np.random.uniform() < cfg.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=cfg.noise_range2, clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=cfg.poisson_scale_range2,
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // cfg.scale, ori_w // cfg.scale), mode=mode)
            out = filter2D(out, sinc_k)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*cfg.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // cfg.scale, ori_w // cfg.scale), mode=mode)
            out = filter2D(out, sinc_k)

        # clamp and round
        lr_batch = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # random crop
        gt_size = cfg.train_img_shape[0]
        gt_batch, lr_batch = paired_random_crop(gt_batch, lr_batch, gt_size, cfg.scale)
        return lr_batch, gt_batch



##### JPEG Compression ##############################

def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality (float): JPG compression quality. 0 for lowest quality, 100 for
            best quality. Default: 90.
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img

def random_add_jpg_compression(img, quality_range=(80, 100)):
    """Randomly add JPG compression artifacts.
    Args:
        img (Numpy array): Input image, shape (h, w, c), range [0, 1], float32.
        quality_range (tuple[float] | list[float]): JPG compression quality
            range. 0 for lowest quality, 100 for best quality.
            Default: (90, 100).
    Returns:
        (Numpy array): Returned image after JPG, shape (h, w, c), range[0, 1],
            float32.
    """
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, quality)


class DegradationKernelGenerator():
    def __init__(self):
        # blur settings for the first degradation
        self.blur_kernel_size = cfg.blur_kernel_size
        self.kernel_list = cfg.kernel_list
        self.kernel_prob = cfg.kernel_prob
        self.blur_sigma = cfg.blur_sigma
        self.betag_range = cfg.betag_range
        self.betap_range = cfg.betap_range
        self.sinc_prob = cfg.sinc_prob

        # blur settings for the second degradation
        self.blur_kernel_size2 = cfg.blur_kernel_size2
        self.kernel_list2 = cfg.kernel_list2
        self.kernel_prob2 = cfg.kernel_prob2
        self.blur_sigma2 = cfg.blur_sigma2
        self.betag_range2 = cfg.betag_range2
        self.betap_range2 = cfg.betap_range2
        self.sinc_prob2 = cfg.sinc_prob2

        # a final sinc filter
        self.final_sinc_prob = cfg.final_sinc_prob

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1
        return

    def get_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < cfg.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < cfg.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < cfg.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_k = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_k = torch.FloatTensor(sinc_k)
        else:
            sinc_k = self.pulse_tensor

        k1 = torch.FloatTensor(kernel)
        k2 = torch.FloatTensor(kernel2)
        return k1,k2,sinc_k