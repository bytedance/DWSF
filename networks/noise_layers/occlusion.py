# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F


class RandomOcclusion(nn.Module):
    """
    overlay non-watermarked image into watermarked image on random position
    """
    def __init__(self, min_ratio=0.125, max_ratio=0.25, target_size=0, proportional=False):
        super(RandomOcclusion, self).__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.proportional = proportional
        self.target_size = target_size

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover

        if self.proportional:
            height_ratio = (np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
            width_ratio = height_ratio
        else:
            height_ratio = (np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)
            width_ratio = (np.random.rand() * (self.max_ratio - self.min_ratio) + self.min_ratio)

        a = list(range(image.shape[0]))
        np.random.shuffle(a)
        paste_img = cover_image[a, :, :, :]
        paste_img = F.interpolate(paste_img, (int(image.shape[2] * height_ratio), int(image.shape[3] * width_ratio)))

        h_start = max(0, np.random.randint(image.shape[2] - paste_img.shape[2]))
        w_start = max(0, np.random.randint(image.shape[3] - paste_img.shape[3]))

        mask = torch.ones_like(image)
        mask[:, :, h_start: h_start + paste_img.shape[2], w_start: w_start + paste_img.shape[3]] = 0
        output = mask * image
        output[:, :, h_start: h_start + paste_img.shape[2], w_start: w_start + paste_img.shape[3]] = paste_img

        if self.target_size != 0:
            output = F.interpolate(output, (self.target_size, self.target_size))

        return output