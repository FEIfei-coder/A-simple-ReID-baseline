from __future__ import absolute_import

from torch.nn import functional as F
from torch import Tensor


def leakyrelu_(z: Tensor, alpha, inplace=False):
	'''
	:param z: the z is the result represented in z = W.t()*x,
			  *: inner product
			  z must be tensor
	:return: max{z, alpha*z}  alpha<0
	'''
	return F.leaky_relu(z, alpha, inplace)
