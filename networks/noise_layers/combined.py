# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/noise_layers/combined.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn
from numpy import random
from .identity import Identity


def get_random_int(int_range):
	return random.randint(int_range[0], int_range[1])


class Combined(nn.Module):
	"""
	A module used to select one of noise layers randomly
	"""
	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Identity()]
		self.list = list

	def forward(self, image_and_cover):
		index = get_random_int([0, len(self.list) - 1])
		return self.list[index](image_and_cover)


class Joint(nn.Module):
	"""
	A module used to pass several noise layers serially
	"""
	def __init__(self, layers):
		super(Joint, self).__init__()
		self.layers = layers

	def forward(self, image_and_cover):
		for layer in self.layers:
			noised_image = layer(image_and_cover)
			image_and_cover = [noised_image, image_and_cover[1]]

		return noised_image