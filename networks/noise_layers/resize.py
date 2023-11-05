# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import torch.nn.functional as F
import numpy as np
from torch import nn


def get_random_rectangle_inside(image_shape, remaining_height, remaining_width):
	image_height = image_shape[2]
	image_width = image_shape[3]

	if remaining_height == image_height:
		height_start = 0
	else:
		if image_height - remaining_height <= 0:
			height_start = np.random.randint(0, 1)
		else:
			height_start = np.random.randint(0, image_height - remaining_height)

	if remaining_width == image_width:
		width_start = 0
	else:
		if image_width - remaining_width <= 0:
			width_start = np.random.randint(0, 1)
		else:
			width_start = np.random.randint(0, image_width - remaining_width)

	return height_start, min(height_start+remaining_height, image_height), width_start, min(width_start + remaining_width, image_width)


class RandomResize(nn.Module):
	"""
	resize image with random sacle ratio
	"""
	def __init__(self, min_ratio=0.5, max_ratio=2, target_size=0, proportional=False):
		super(RandomResize, self).__init__()
		self.proportional = proportional
		self.min_ratio = min_ratio
		self.max_ratio = max_ratio
		self.target_size = target_size

	def random_ratio(self, resize_to_small_or_big):
		if resize_to_small_or_big == 0:
			ratio = (np.random.rand() * (self.max_ratio-1) + 1) 
		else:
			ratio = (np.random.rand() * (1 - self.min_ratio) + self.min_ratio)
		
		return ratio

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		image_height = image.shape[2]
		image_width = image.shape[3]

		if self.proportional:
			resize_to_small_or_big = np.random.randint(2)
			height_ratio = self.random_ratio(resize_to_small_or_big)
			width_ratio = height_ratio
		else:
			resize_to_small_or_big = np.random.randint(2)
			height_ratio = self.random_ratio(resize_to_small_or_big)
			resize_to_small_or_big = np.random.randint(2)
			width_ratio = self.random_ratio(resize_to_small_or_big)


		resize_height = int(height_ratio * image_height)
		resize_width = int(width_ratio * image_width)
		
		output = F.interpolate(image, size=(resize_height, resize_width), mode='bilinear')

		if self.target_size != 0:
			output = F.interpolate(output,(self.target_size, self.target_size))

		return output
