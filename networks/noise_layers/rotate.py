# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Inc. and/or its affiliates
# ------------------------------------------------------------------------
import numpy as np
from torch import nn
import torchvision.transforms.functional as F2


class RandomRotate(nn.Module):
	"""
	rotate image with random angle
	"""
	def __init__(self, angle=30):
		super(RandomRotate, self).__init__()
		self.angle = angle

	def forward(self, image_and_cover):
		image, cover_image = image_and_cover
		angle = np.random.randint(-self.angle, self.angle)
		output = F2.rotate(image, angle)

		return output