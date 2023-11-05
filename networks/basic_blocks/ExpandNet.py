# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/blocks/ExpandNet.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn


class ConvTBNRelu(nn.Module):
	"""
	A sequence of TConvolution, Batch Normalization, and ReLU activation
	"""
	def __init__(self, channels_in, channels_out, stride=2, dilation=1, groups=1):
		super(ConvTBNRelu, self).__init__()

		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(True)
		if stride == 1:
			kernel_size = 3
			padding = 1
		elif stride == 2:
			kernel_size = 2
			padding = 0
		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups),
			Normalize(channels_out),
			Activation
		)

	def forward(self, x):
		return self.layers(x)


class ExpandNet(nn.Module):
	"""
	Network that composed by layers of ConvTBNRelu
	"""
	def __init__(self, in_channels, out_channels, blocks, stride=2, dilation=1, groups=1):
		super(ExpandNet, self).__init__()

		layers = [ConvTBNRelu(in_channels, out_channels, stride=stride, dilation=dilation, groups=groups)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvTBNRelu(out_channels, out_channels, stride=stride, dilation=dilation, groups=groups)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
