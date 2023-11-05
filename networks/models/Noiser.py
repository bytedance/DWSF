# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/Noise.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from .. import *
from ..noise_layers import *
from ..noise_layers.crop import RandomCrop
from ..noise_layers.dropout import RandomDropout
from ..noise_layers.gaussian_filter import RandomGF
from ..noise_layers.gaussian_noise import RandomGN
from ..noise_layers.identity import Identity
from ..noise_layers.jpeg import RandomJpegTest, RandomJpegMask, RandomJpegSS, RandomJpeg
from ..noise_layers.resize import RandomResize
from ..noise_layers.rotate import RandomRotate
from ..noise_layers.pip import RandomPIP
from ..noise_layers.occlusion import RandomOcclusion
from ..noise_layers.color import RandomColor
from ..noise_layers.padding import RandomPadding
from ..noise_layers.combined import Combined,Joint
from torch import nn


class Noise(nn.Module):
    """
    A Noise Network
    """
    def __init__(self, layers):
        super(Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)

    def forward(self, image_and_cover):
        noised_image = self.noise(image_and_cover)
        return noised_image

