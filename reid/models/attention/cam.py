from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor



class ChannelAttention_(nn.Module):
	def __init__(self, channel: int, ratio: int=16):
		'''
		:param channel: channel of the img
		:param ratio: the re-scale ratio
		'''
		super(ChannelAttention_, self).__init__()
		self.squeeze = nn.AdaptiveAvgPool2d(1)
		self.excitation = nn.Sequential(
			nn.Linear(in_features=channel, out_features=channel // ratio),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=channel // ratio, out_features=channel),
			nn.Sigmoid()
		)

	def forward(self, x: Tensor):
		'''
		:param x: feature map
		:return: the weight vecter of each channel
		'''
		b, c, _, _ = x.size()
		y = self.squeeze(x).view(b, c)
		z = self.excitation(y).view(b, c, 1, 1)
		# return x * z.expand_as(x)
		return z.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
