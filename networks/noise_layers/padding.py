# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class RandomPadding(nn.Module):
    """
    pad image with random padding size
    """
    def __init__(self, padding_min=0, padding_max=100, target_size=0):
        super(RandomPadding, self).__init__()
        self.min_size = padding_min
        self.max_size = padding_max
        self.target_size = target_size

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        
        left = np.random.randint(self.min_size, self.max_size)
        right = np.random.randint(self.min_size, self.max_size)
        up = np.random.randint(self.min_size, self.max_size)
        down = np.random.randint(self.min_size, self.max_size)
        mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]+up+down,image.shape[3]+left+right)).to(image.device)

        mask[:, :, up:mask.shape[2]-down, left:mask.shape[3]-right] = image

        if self.target_size != 0:
            mask = F.interpolate(mask,(self.target_size, self.target_size)) 
        
        return mask


