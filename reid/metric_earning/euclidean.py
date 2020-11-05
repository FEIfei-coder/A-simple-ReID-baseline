from __future__ import absolute_import

import torch.nn.functional as F
from torch import Tensor


def euclidean_(x: Tensor, y: Tensor, p: float, keepdim: bool=True):
	'''
	:param x: input tensor
	:param y: input tensor
	:param p: p norm
	:param keepdim: dafault False
	:return: dist between x & y
	'''
	return F.pairwise_distance(x1=x, x2=y, p=p, keepdim=keepdim)
