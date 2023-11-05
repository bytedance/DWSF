# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/noise_layers/gaussian_noise.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
import numpy as np
import torch
import torch.nn as nn


class RandomGN(nn.Module):
	"""
	add gaussian noise with random var
	"""
	def __init__(self, min_var=3, max_var=10, mean=0):
		super(RandomGN, self).__init__()
		self.min_var = min_var
		self.max_val = max_var
		self.mean = mean

	def gaussian_noise(self, image, mean, var):
		noise = torch.Tensor(np.random.normal(mean, var**0.5, image.shape)/128.).to(image.device)
		out = image + noise
		return out

	def forward(self, image_and_cover):
		self.var = np.random.rand() * (self.max_val - self.min_var) + self.min_var
		image, cover_image = image_and_cover
		return self.gaussian_noise(image, self.mean, self.var)