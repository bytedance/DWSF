# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/Encoder_MP.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from ..basic_blocks.ConvNet import ConvBNRelu
from ..basic_blocks.ExpandNet import ExpandNet
from ..basic_blocks.SENet import SENet, SENet_decoder
import torch
import numpy as np
from torch import nn


class Encoder(nn.Module):
    """
    Encoder for proposed method
    """
    def __init__(self, H=128, W=128, message_length=30, encoder_channels=64, encoder_blocks=4, in_channel=3, out_channel=3):
        super(Encoder, self).__init__()
        self.H = H
        self.W = W
        self.conv_channels = encoder_channels
        self.num_blocks = encoder_blocks
        self.message_size = int(encoder_channels ** 0.5)
        self.expandblock = int(np.log2(self.H // self.message_size))

        self.linear = nn.Linear(message_length, encoder_channels)
        self.message_preprocess = nn.Sequential(
            ConvBNRelu(1, encoder_channels),
            ExpandNet(encoder_channels, encoder_channels, blocks=self.expandblock),
            SENet(encoder_channels, encoder_channels, blocks=2)
        )

        self.image_preprocess = nn.Sequential(
            ConvBNRelu(in_channel, self.conv_channels),
            SENet(self.conv_channels, self.conv_channels, blocks=encoder_blocks)
        )

        self.after_concat_layer1 = nn.Sequential(
            SENet(self.conv_channels + encoder_channels, self.conv_channels + encoder_channels, blocks=1),
            SENet_decoder(self.conv_channels + encoder_channels, self.conv_channels ,blocks=2,drop_rate2=1),
            SENet(self.conv_channels, self.conv_channels, blocks=encoder_blocks),
            ExpandNet(self.conv_channels, self.conv_channels, blocks=1)
        )

        self.after_concat_layer2 = nn.Sequential(
            SENet(self.conv_channels + encoder_channels, self.conv_channels + encoder_channels, blocks=1),
            SENet_decoder(self.conv_channels + encoder_channels, self.conv_channels,blocks=2,drop_rate2=1),
            SENet(self.conv_channels, self.conv_channels, blocks=encoder_blocks),
            ExpandNet(self.conv_channels, self.conv_channels, blocks=1)
        )

        self.after_concat_layer3 = nn.Sequential(
            SENet(self.conv_channels + encoder_channels, self.conv_channels + encoder_channels, blocks=1),
            SENet_decoder(self.conv_channels + encoder_channels, self.conv_channels,blocks=2,drop_rate2=1),
            SENet(self.conv_channels, self.conv_channels, blocks=encoder_blocks),
            ExpandNet(self.conv_channels, self.conv_channels, blocks=1)
        )

        self.after_concat_layer4 = nn.Sequential(
            ConvBNRelu(self.conv_channels + in_channel, self.conv_channels),
            SENet(self.conv_channels, self.conv_channels, blocks=encoder_blocks),
            nn.Conv2d(self.conv_channels, out_channel, kernel_size=1)
        )

        self.final_layer = nn.Conv2d(out_channel+in_channel, out_channel, kernel_size=1)
    
    def forward(self, image, message):

        message_duplicate = self.linear(message)
        message_image = message_duplicate.view(-1, 1, self.message_size, self.message_size)
        message_pre = self.message_preprocess(message_image)

        x = self.image_preprocess(image)

        concat1 = torch.cat([message_pre, x], dim=1)
        x = self.after_concat_layer1(concat1)

        concat2 = torch.cat([message_pre, x], dim=1)
        x = self.after_concat_layer2(concat2)

        concat3 = torch.cat([message_pre, x], dim=1)
        x = self.after_concat_layer3(concat3)

        concat4 = torch.cat([image, x], dim=1)
        x = self.after_concat_layer4(concat4)

        concat5 = torch.cat([image, x], dim=1)
        x = self.final_layer(concat5)

        return x

