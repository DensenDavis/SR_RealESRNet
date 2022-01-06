
import torch 
import os
from datetime import datetime


class Configuration():
    def __init__(self):
        #general config

        # dataset config
        self.train_gt_path = 'ds/train/hr/*.jpg'
        self.val_gt_path = 'ds/val/hr/*.jpg'
        self.val_input_path = 'ds/val/lr/*.jpg'
        self.train_batch_size = 2
        self.val_batch_size = 2
        self.train_img_shape = [256,256]
        self.train_augmentation = True
        self.val_img_shape = [512,512]
        self.val_augmentation = False

        # self.losses = {"l1_loss", "vgg_loss"}
        self.loss_weights = {"l1_loss":1, "vgg_loss":1}

        self.ckpt_dir = None
        self.train_mode = ['best', 'last'][1]
        self.n_epochs = 10000
        self.val_freq = 1
        self.display_frequency = 2
        self.display_samples = 4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log_dir = os.path.join('logs', str(datetime.now().strftime("%d%m%Y-%H%M%S")))  # Tensorboard logging

        self.scale = 2

        self.blur_kernel_size = 7
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]
        self.sinc_prob = 0.1

        # blur settings for the second degradation
        self.blur_kernel_size2 = 7
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.blur_sigma2 = [0.2, 3]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]
        self.sinc_prob2 = 0.1

        # a final sinc filter
        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 7)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1


        # the first degradation process
        self.resize_prob = [0.2, 0.7, 0.1]  # up, down, keep
        self.resize_range = [0.15, 1.5]
        self.gaussian_noise_prob = 0.5
        self.noise_range = [1, 30]
        self.poisson_scale_range = [0.05, 3]
        self.gray_noise_prob = 0.4
        self.jpeg_range = [30, 95]

        # the second degradation process
        self.second_blur_prob = 0.8
        self.resize_prob2 = [0.3, 0.4, 0.3]  # up, down, keep
        self.resize_range2 = [0.3, 1.2]
        self.gaussian_noise_prob2 = 0.5
        self.noise_range2 = [1, 25]
        self.poisson_scale_range2 = [0.05, 2.5]
        self.gray_noise_prob2 = 0.4
        self.jpeg_range2 = [30, 95]