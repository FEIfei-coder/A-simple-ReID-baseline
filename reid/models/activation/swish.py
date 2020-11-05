from __future__ import absolute_import

import torch
from torch.nn import functional as F
from torch import Tensor


def swish_(z: Tensor, inplace=False):
	'''
	:param z: the z is the result represented in z = W.t()*x,
			  *: inner product
			  z must be tensor
	:return: x * sigmoid(x)
	'''
	return z * torch.sigmoid(z)
