# Code ported from https://github.com/jzyustc/MBRS/blob/main/network/Encoder_MP_Decoder.py
# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2023) B-
# ytedance Inc..
# *************************************************************************
from torch import nn
from .Encoder import Encoder
from .Decoder import Decoder
from .Noiser import Noise


class EncoderDecoder(nn.Module):
	"""
	A Sequential of Encoder-Noise-Decoder
	"""
	def __init__(self, H, W, message_length, noise_layers, blocks=4):
		super(EncoderDecoder, self).__init__()

		self.encoder = Encoder(H=H, W=W, message_length=message_length, encoder_blocks=blocks)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder(message_length=message_length)

	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)

		return encoded_image, noised_image, decoded_message


