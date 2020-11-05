from __future__ import absolute_import

import torch.nn as nn
from torch import Tensor


class SpatialAttention_(nn.Module):
	'''
	the spatial attention mechanism we define
	'''
	def __init__(self, c: int):
		'''
		:param c: channel of the img
		'''
		super(SpatialAttention_, self).__init__()
		self.channel_input = c
		self.squeeze = nn.Conv2d(in_channels=self.channel_input, out_channels=1, kernel_size=1)
		self.excitation = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
			nn.Sigmoid()
		)


	def forward(self, inputs: Tensor):
		b, c, h, w = inputs.size()
		y = self.squeeze(inputs)
		z = self.excitation(y)

		return z.expand_as(inputs)


class SpatialAttention(nn.Module):
	def __init__(self):
		super(SpatialAttention, self).__init__()


	def forward(self):
		pass




