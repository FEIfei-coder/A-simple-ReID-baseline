from __future__ import absolute_import

import torch.nn as nn
from torch import Tensor



class BAM_Block(nn.Module):
	'''
	Attention module in series
	you can also write the class by import sam & import cam

	we achieve this module which is from: https://arxiv.org/pdf/1807.06514.pdf
	'''
	def __init__(self, c: int, ratio: int):
		'''
		:param channel: channel of the img
		:param ratio: the re-scale ratio
		'''
		super(BAM_Block, self).__init__()
		self.cam_squeeze = nn.AdaptiveAvgPool2d(1)
		self.sam_squeeze = nn.Conv2d(in_channels=c, out_channels=c // ratio, kernel_size=1)

		self.cam_excitation = nn.Sequential(
			# nn.ReLU(inplace=True),
			nn.Linear(in_features=c, out_features=c // ratio),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=c // ratio, out_features=c)
		)
		self.sam_excitation = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=c // ratio, out_channels=1, kernel_size=3, stride=1, padding=5, dilation=4),
		)

		self.non_linear = nn.Sigmoid()


	def forward(self, x: Tensor):
		b, c, h, w = x.size()

		y_cam = self.cam_squeeze(x).view(b, c)
		y_sam = self.sam_squeeze(x)

		z_cam = self.cam_excitation(y_cam).view(b, c, 1, 1)
		z_sam = self.sam_excitation(y_sam)

		return self.non_linear(z_cam.expand_as(x) + z_sam.expand_as(x))

