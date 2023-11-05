# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop, ToPILImage, Pad, RandomRotation
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision.transforms.functional as F2
import glob


class EdDataset(Dataset):
    """
    Dataset for encoder-decoder
    """
    def __init__(self, filepath, transform):
        self.filepath = filepath
        self.filelist = os.listdir(self.filepath)
        self.tranform = transform

    def __getitem__(self, item):
        image = Image.open(self.filepath+'/'+self.filelist[item])
        image = self.tranform(image)
        return image

    def __len__(self):
        return len(self.filelist)


class RandomGaussianBlurPadding(object):
    def __init__(self):
        self.sigma = 3

    def __call__(self, img):
        kernel = np.random.randint(3, 8) // 2 * 2 + 1
        img = transforms.GaussianBlur(kernel_size=kernel, sigma=3)(img)
        return img


class RandomColorPadding(object):
    def __init__(self):
        self.minr = 0.5
        self.maxr = 1.5

    def __call__(self, image):
        choice = np.random.randint(0, 3)
        r = np.random.uniform(self.minr, self.maxr)
        if choice == 0:
            image = F2.adjust_brightness(image, r)
        elif choice == 1:
            image = F2.adjust_contrast(image, r)
        elif choice == 2:
            image = F2.adjust_saturation(image, r)
        return image


class RandomJPEGPadding(object):
    def __init__(self, wm_path, tmp_path, quality_factor):
        self.wm_path = wm_path
        self.quality_factor = quality_factor
        self.tmp_path = tmp_path
        self.T = ToTensor()
        self.P = ToPILImage()
        if os.path.exists(self.tmp_path) == False:
            os.makedirs(self.tmp_path)

    def __call__(self, img):
        tmp_name = os.path.join(self.tmp_path, self.wm_path.split('/')[-1])
        save_path = '{}.jpeg'.format(tmp_name.split('.')[0])
        img = self.P(img)
        img.save(save_path, quality=self.quality_factor, subsampling=2)
        img = Image.open(save_path).convert('RGB')
        os.remove(save_path)
        return self.T(img)


class RandomNoisePadding(object):
    def __init__(self):
        pass
    def __call__(self, tensor_img):
        var = np.random.rand() * (10 - 3) + 3
        noise = torch.Tensor(np.random.normal(0, var ** 0.5, tensor_img.shape) / 128.)
        tensor_img = tensor_img + noise
        del noise
        return torch.clamp(tensor_img, 0, 1)


class RandomRotatePadding:
    def __init__(self, angle):
        self.Rotate = RandomRotation(angle)

    def __call__(self, img):
        return self.Rotate(img)


class RandomResizePadding:
    def __init__(self, height_ratio, width_ratio):
        self.resize_height = int(height_ratio * 512)
        self.resize_width = int(width_ratio * 512)
        self.Pad = Pad((0, 0, max(0, 512 - self.resize_width), max(0, 512 - self.resize_height)), fill=0,padding_mode='constant')

    def __call__(self, tensor_img):
        img = F.interpolate(tensor_img.unsqueeze(0), size=(self.resize_height, self.resize_width), mode='bilinear')
        img = self.Pad(img.squeeze(0))[:, :512, :512]

        return img


class SegDataset(torch.utils.data.Dataset):
    """
    Dataset for segmentation model
    """
    def __init__(self, path_wm=None, path_mask=None, tmp_path=None):
        self.wm_path = glob.glob(os.path.join(path_wm, '*'))
        self.tmp_path = tmp_path
        self.path_mask = path_mask
        self.totensor = ToTensor()

    def random_ratio(self, resize_to_small_or_big):
        if resize_to_small_or_big == 0:
            ratio = (np.random.rand() * 1 + 1)
        else:
            ratio = (np.random.rand() * 0.5 + 0.5)
        return ratio

    def __len__(self):
        return len(self.wm_path)

    def __getitem__(self, index):
        wm_path = self.wm_path[index]
        split_name = wm_path.split('/')[-1]
        mask_path = os.path.join(self.path_mask, split_name)
        wm_image = Image.open(wm_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')

        wm_image = self.totensor(wm_image)
        mask_image = self.totensor(mask_image)

        quality_factor = random.randint(50, 90)

        resize_to_small_or_big = np.random.randint(2)
        scale_factor1 = self.random_ratio(resize_to_small_or_big)
        resize_to_small_or_big = np.random.randint(2)
        scale_factor2 = self.random_ratio(resize_to_small_or_big)

        transform_map = {
            "2": RandomRotatePadding(40),
            "3": RandomCrop((512, 512)),
            "4": RandomResizePadding(scale_factor1, scale_factor2),
            "6": RandomNoisePadding(),
            "7": RandomColorPadding(),
            "8": RandomGaussianBlurPadding(),
            "9": RandomJPEGPadding(wm_path=wm_path, tmp_path=self.tmp_path, quality_factor=quality_factor)}

        transform_list = [random.choice([["4","9"], "2", "3", "4", "6", "7", "8", "9", "2", "9", "2", "2"]),"3"]
        transform_list_flatten = [i for item in transform_list for i in item]
        img_transform = T.Compose([transform_map[key] for key in transform_list_flatten])
        mask_transform = T.Compose([transform_map[key] for key in transform_list_flatten if key <= "5"])

        seed = random.randint(0, 99999999)
        random.seed(seed)
        torch.manual_seed(seed)
        wm_image = img_transform(wm_image)
        random.seed(seed)
        torch.manual_seed(seed)
        mask_image = mask_transform(mask_image)
        wm_image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(wm_image)

        return wm_image, mask_image, wm_path.split('/')[-1], "_".join([key for key in transform_list_flatten]) + '_{}'.format(quality_factor)
