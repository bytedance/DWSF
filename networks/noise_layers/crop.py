# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/noise_layers/crop.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn
import numpy as np
import torch.nn.functional as F

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height, image_width = image_shape[2], image_shape[3]
	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class RandomCrop(nn.Module):
	"""
	crop image randomly
	"""
	def __init__(self, min_ratio=0.7, max_ratio=1, target_size=0, proportional=False):
		super(RandomCrop, self).__init__()
		self.min_ratio = min_ratio
		self.max_ratio = max_ratio
		self.proportional = proportional
		self.target_size = target_size

	def forward(self, image_and_cover):
		if self.proportional:
			height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio) 
			width_ratio = height_ratio
		else:
			height_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
			width_ratio = (np.random.rand() * (self.max_ratio-self.min_ratio) + self.min_ratio)
		
		image, cover_image = image_and_cover

		h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, height_ratio, width_ratio)
		output = image[:, :, h_start: h_end, w_start: w_end]

		if self.target_size !=0:
			output = F.interpolate(output, (self.target_size, self.target_size))
		return output










