from __future__ import absolute_import

from torch.nn import functional as F
from torch import Tensor


def softplus_(z: Tensor):
	'''
	:param input: param z: the z is the result represented in z = W.t()*x,
			      *: inner product
			  	  z must be tensor
	:return: y = ln(1+exp(z))
	'''
	return F.softplus(z)

