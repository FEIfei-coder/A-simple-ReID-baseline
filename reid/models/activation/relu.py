from __future__ import absolute_import

from torch.nn import functional as F
from torch import Tensor


def relu_(z: Tensor, inplace=False):
	'''
	:param z: the z is the result represented in z = W.t()*x,
			  *: inner product
			  z must be tensor
	:return: max{z, 0}
	'''
	return F.relu(z, inplace)
