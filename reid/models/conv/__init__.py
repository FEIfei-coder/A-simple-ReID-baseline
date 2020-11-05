from __future__ import absolute_import

import torch
import torch.nn as nn


class Conv(nn.Module):
	'''
	The class inherits from the nn.Module to manage the conv mechanism we override,
	and packaging them
	'''

	__factory = {

	}

	def __init__(self, name, size=3, stride=1, padding=0):
		'''
		:param z: the z is the result represented in z = W.t()*x, *:inner product
		'''
		super(Conv, self).__init__()
		self.name = name
		self.size = size
		self.stride = stride
		self.padding = padding



	def forward(self, input_feature_map, *args):
		if self.name not in self.__factory.keys():
			raise KeyError("Unknown convolution module: {}".format(self.name))
		return self.__factory[self.name](self.size, self.stride, self.padding, *args)


	def get_names(self):
		return self.__factory.keys()
