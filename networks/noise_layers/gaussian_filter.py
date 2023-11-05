# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/noise_layers/gaussian_filter.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
import torch.nn as nn
from kornia.filters import GaussianBlur2d
import numpy as np


class RandomGF(nn.Module):
	"""
	blur image with random kernel size
	"""
	def __init__(self, min_kernel=3, max_kernel=8, sigma=3):
		super(RandomGF, self).__init__()
		self.min_kernel = min_kernel
		self.max_kernel = max_kernel
		self.sigma = sigma

	def forward(self, image_and_cover):
		kernel = np.random.randint(self.min_kernel, self.max_kernel)//2*2+1
		self.gaussian_filter = GaussianBlur2d((kernel, kernel), (self.sigma, self.sigma))
		image, cover_image = image_and_cover

		return self.gaussian_filter(image)