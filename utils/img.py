# *************************************************************************
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Code ported from https://github.com/facebookresearch/ssl_watermarking/blob/main/utils_img.py
# *************************************************************************
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
import numpy as np
import torch
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
image_mean = torch.Tensor(NORMALIZE_IMAGENET.mean).view(-1, 1, 1).to(device)
image_std = torch.Tensor(NORMALIZE_IMAGENET.std).view(-1, 1, 1).to(device)


def psnr_clip(x, y, target_psnr):
    """ 
    Clip x so that PSNR(x,y)=target_psnr 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
        target_psnr: Target PSNR value in dB
    """
    delta = x - y
    delta = 255 * (delta * image_std.to(x.device))
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    # if psnr<target_psnr:
    #     delta = (torch.sqrt(10**((psnr-target_psnr)/10))) * delta
    delta = (torch.sqrt(10**((psnr-target_psnr)/10))) * delta

    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2))
    delta = (delta / 255.0) / image_std.to(x.device)
    return y + delta


def draw_mask(height, width, h_coor, w_coor, splitSize, image=None):
    if image is None:
        image = np.zeros((height,width))
    for i in range(len(h_coor)):
        image[h_coor[i]-splitSize//2:h_coor[i]+splitSize//2,w_coor[i]-splitSize//2:w_coor[i]+splitSize//2] = 255
    return image