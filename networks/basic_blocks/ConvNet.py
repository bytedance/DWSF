# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/blocks/ConvNet.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn


class ConvBNRelu(nn.Module):
	"""
	A sequence of Convolution, Batch Normalization, and ReLU activation
	"""
	def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, dilation=1, groups=1, padding=1):
		super(ConvBNRelu, self).__init__()
		
		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(True)
		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, kernel_size, stride, padding=padding, dilation=dilation, groups=groups),
			Normalize(channels_out),
			Activation
		)

	def forward(self, x):
		return self.layers(x)


class ConvNet(nn.Module):
	"""
	Network that composed by layers of ConvBNRelu
	"""
	def __init__(self, in_channels, out_channels, blocks, stride=1, dilation=1, groups=1, padding=1):
		super(ConvNet, self).__init__()

		layers = [ConvBNRelu(in_channels, out_channels, stride=stride, dilation=dilation, groups=groups, padding=padding)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvBNRelu(out_channels, out_channels, stride=stride, dilation=dilation, groups=groups, padding=padding)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)



