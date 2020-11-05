from __future__ import absolute_import

import torch
from torch import Tensor


def tanh_(z: Tensor):
	'''
	:param z: the z is the result represented in z = W.t()*x,
			  *: inner product
			  z must be tensor
	:return:
	'''
	return torch.tanh(z)
