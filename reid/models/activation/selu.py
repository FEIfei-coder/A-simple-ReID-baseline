from __future__ import absolute_import

from torch.nn import functional as F
from torch import Tensor

# def selu_(z, gamma=2, alpha=1, inplace=False):
# 	'''
# 	:param z: the z is the result represented in z = W.t()*x,
# 			  *: inner product
# 			  z must be tensor
# 	:return: gamma * max{z, alpha(exp(z)-1)}
# 	'''
# 	return  gamma * F.elu(z, alpha, inplace)


def selu_(z: Tensor, inplace=False):
	return F.selu(z, inplace)


