from __future__ import absolute_import

import torch
from torch import Tensor


def norm_(x: Tensor, p: str, dim: int, keepdim: bool):
	'''
	:param x: input Tensor
	:param p: int, float, inf, -inf, 'fro', 'nuc', optional
	:param dim: dim
	:param keepdim: Ture / False
	:return: the normed value in you set the dim
	'''
	return torch.norm(x, p, dim, keepdim)
