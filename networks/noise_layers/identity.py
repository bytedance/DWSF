# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/noise_layers/identity.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn


class Identity(nn.Module):
	"""
	Identity-mapping noise layer. Does not change the image
	"""
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		return image
