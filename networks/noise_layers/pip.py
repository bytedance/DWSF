# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch.nn.functional as F


def get_random_rectangle_inside(image_shape, image_shape2):
    image_height = image_shape[2]
    image_width = image_shape[3]

    remaining_height = image_shape2[2]
    remaining_width = image_shape2[3]

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height + 1)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width + 1)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class RandomPIP(nn.Module):
    """
    overlay watermarked image into non-watermarked image on random position
    """
    def __init__(self, min_ratio=1, max_ratio=2, target_size=0, proportional=False):
        super(RandomPIP, self).__init__()
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.proportional = proportional
        self.target_size = target_size

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        
        if self.proportional == True:
            height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio) 
            width_ratio = height_ratio
        else:
            height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
            width_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)

        a = list(range(image.shape[0]))
        np.random.shuffle(a)
        bg_image = cover_image[a, :, :, :]
        bg_image = F.interpolate(bg_image, (int(image.shape[2]*height_ratio), int(image.shape[3]*width_ratio)))

        h_start = np.random.randint(max(1, bg_image.shape[2]-image.shape[2]))
        w_start = np.random.randint(max(1, bg_image.shape[3]-image.shape[3]))
        
        bg_image[:, :, h_start: h_start+image.shape[2], w_start: w_start+image.shape[3]] = image
        
        if self.target_size != 0:
            bg_image = F.interpolate(bg_image, (self.target_size, self.target_size))

        return bg_image
 
