from __future__ import absolute_import

import torch.nn as nn
from torch import Tensor


class CBAM_Block(nn.Module):
	'''
	Attention module in parallel
	you can also write the class by import sam & import cam

	we achieve this module which is from: https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf
	'''
	def __init__(self, c: int, ratio: int):
		'''
		:param channel: channel of the img
		:param ratio: the re-scale ratio
		'''
		super(CBAM_Block, self).__init__()

		self.cam_avg_squeeze = nn.AdaptiveAvgPool2d(1)
		self.cam_max_squeeze = nn.AdaptiveMaxPool2d(1)

		self.sam_squeeze = nn.Conv2d(in_channels=c, out_channels=c // ratio, kernel_size=1)
		# self.sam = nn.AvgPool2d
		# self.sam = nn.MaxPool2d

		self.cam_excitation = nn.Sequential(
			# nn.ReLU(inplace=True),
			nn.Linear(in_features=c, out_features=c // ratio),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=c // ratio, out_features=c)
		)
		self.sam_excitation = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=c // ratio, out_channels=1, kernel_size=3, padding=1),
		)

		self.non_linear = nn.Sigmoid()


	def forward(self, x: Tensor):
		b, c, h, w = x.size()

		y_avg_cam = self.cam_avg_squeeze(x).view(b, c)
		y_max_cam = self.cam_max_squeeze(x).view(b, c)
		z_avg_cam = self.cam_excitation(y_avg_cam).view(b, c, 1, 1)
		z_max_cam = self.cam_excitation(y_max_cam).view(b, c, 1, 1)
		z_avg_cam = z_avg_cam.expand_as(x)
		z_max_cam = z_max_cam.expand_as(x)
		z_cam = self.non_linear(z_avg_cam+z_max_cam)

		y_sam = self.sam_squeeze(z_cam)
		z_sam = self.sam_excitation(y_sam)
		return self.non_linear(z_sam.expand_as(x))


