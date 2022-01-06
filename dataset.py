import os
import math
import glob
import PIL
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from degradation import random_add_jpg_compression, DegradationKernelGenerator
from config import Configuration
cfg = Configuration()

class TrainDataset(Dataset):
    def __init__(self, gt_folder):
        self.gt_files = glob.glob(gt_folder)
        self.num_batches = math.ceil(len(self.gt_files)/cfg.train_batch_size)
        self.transforms_list = [transforms.InterpolationMode.BILINEAR,transforms.InterpolationMode.BICUBIC]
        self.alias_cond = [True,False]
        self.degradation_kernels = DegradationKernelGenerator()
        self.image_transform = transforms.Compose([
            transforms.RandomCrop(cfg.train_img_shape[0]+200, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()            
        ])
        self.tensor_convert = transforms.ToTensor()
        self.image_convert = transforms.ToPILImage()

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, item):
        img = self.gt_files[item]
        img = PIL.Image.open(img)
        img = self.image_transform(img)
        gt_img = self.tensor_convert(img)
        k1,k2,sinc_k = self.degradation_kernels.get_kernels()
        return {'gt_img': gt_img,"k1":k1,"k2":k2,"sinc_k":sinc_k}

class ValDataset(Dataset):
    def __init__(self, input_folder, gt_folder):
        self.gt_files = glob.glob(gt_folder)
        self.num_batches = math.ceil(len(self.gt_files)/cfg.val_batch_size)
        self.input_files = glob.glob(input_folder)
        self.tensor_convert = transforms.ToTensor()
        self.image_convert = transforms.ToPILImage()
        self.gt_image_transform = transforms.Compose([
            transforms.CenterCrop(cfg.val_img_shape)])
        self.input_image_transform = transforms.Compose([
            transforms.CenterCrop((cfg.val_img_shape[0]//2,cfg.val_img_shape[1]//2))])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, item):
        input_img = self.input_files[item]
        gt_img = self.gt_files[item]
        input_img = PIL.Image.open(input_img).convert('RGB')
        gt_img = PIL.Image.open(gt_img).convert('RGB')
        input_img = self.input_image_transform(input_img)
        gt_img = self.gt_image_transform(gt_img)
        input_img = self.tensor_convert(input_img)
        gt_img = self.tensor_convert(gt_img)

        return {'input_img': input_img,'gt_img': gt_img}

class Dataset():
    def __init__(self):
        self.train_ds = DataLoader(TrainDataset(cfg.train_gt_path), batch_size=cfg.train_batch_size, shuffle=True,num_workers=os.cpu_count())
        self.val_ds = DataLoader(ValDataset(cfg.val_input_path,cfg.val_gt_path), batch_size=cfg.val_batch_size, shuffle=True,num_workers=os.cpu_count())
        return
