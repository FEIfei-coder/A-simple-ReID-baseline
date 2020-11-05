from __future__ import absolute_import

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from .softplus import *
from .mish import *
from .sigmoid import *
from .relu import *
from .tanh import *
from .leakyrelu import *
from .elu import *
from .maxout import *
from .swish import *
from .selu import *


class Activate(nn.Module):
	'''
	The class inherits from the nn.Module to manage the activation we override,
	and packaging them likes Non-linear Activations in torch
	'''

	__factory = {
		'softplus': softplus_,
		'mish': mish_,
		'sigmoid': sigmoid_,
		'relu': relu_,
		'leakrelu': leakyrelu_,
		'tanh': tanh_,
		'elu': elu_,
		'swish': swish_,
		'selu': selu_
	}

	def __init__(self, name):
		'''
		:param z: the z is the result represented in z = W.t()*x, *:inner product
		'''
		super(Activate, self).__init__()
		self.name = name


	def forward(self, z, *args):
		if self.name not in self.__factory.keys():
			raise KeyError("Unknown activation: {}".format(self.name))
		return self.__factory[self.name](z, *args)


	def get_names(self):
		return self.__factory.keys()


# if __name__ == '__main__':
# 	x = torch.linspace(-10,10,1000)
# 	act = Activate('elu')
# 	y = act(x)
#
# 	plt.plot(x,y)
# 	plt.grid()
# 	plt.show()


