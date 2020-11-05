from __future__ import absolute_import

from torch.nn import functional as F

from torch import Tensor

def elu_(z: Tensor, alpha=1, inplace=False):
	'''
	:param z: the z is the result represented in z = W.t()*x,
			  *: inner product
			  z must be tensor
	:return: max{z, alpha(exp(z)-1)}
	'''
	return F.elu(z, alpha, inplace)
