# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/blocks/SENet.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
	"""
	A Bottleneck Block
	"""
	def __init__(self, in_channels, out_channels, r, drop_rate=1, groups=1, dilation=1, se=True):
		super(BottleneckBlock, self).__init__()

		self.downsample = None
		self.se = se

		Normalize = nn.BatchNorm2d
		Activation = nn.ReLU(True)

		if (drop_rate == 2) or (in_channels != out_channels):
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
						  stride=drop_rate, groups=groups, dilation=dilation, bias=False),
				Normalize(out_channels)
			)

		self.left = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
					    stride=drop_rate, groups=groups, dilation=dilation, padding=0, bias=False),
			Normalize(out_channels),
			Activation,
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, 
						groups=groups, dilation=dilation, bias=False),
			Normalize(out_channels),
			Activation,
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, 
						groups=groups, dilation=dilation, bias=False),
			Normalize(out_channels)
		)

		self.se = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		identity = x
		x = self.left(x)

		if self.se:
			scale = self.se(x)
			x = x * scale

		if self.downsample is not None:
			identity = self.downsample(identity)

		x += identity
		x = F.relu(x)

		return x


class SENet(nn.Module):
	"""
	SENet with BottleneckBlock
	"""
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, groups=1, dilation=1, drop_rate=1, se=True):
		super(SENet, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = eval(block_type)(out_channels, out_channels, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class SENet_decoder(nn.Module):
	"""
	SENet with BottleneckBlock
	"""
	def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, groups=1, dilation=1, drop_rate=2, drop_rate2=2, se=True):
		super(SENet_decoder, self).__init__()

		layers = [eval(block_type)(in_channels, out_channels, r=r, drop_rate=1, se=se, groups=groups, dilation=dilation)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer1 = eval(block_type)(out_channels, out_channels, r=r, drop_rate=1, se=se, groups=groups, dilation=dilation)
			layers.append(layer1)
			layer2 = eval(block_type)(out_channels, out_channels * drop_rate2, r=r, drop_rate=drop_rate, se=se, groups=groups, dilation=dilation)
			out_channels *= drop_rate2
			layers.append(layer2)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)
