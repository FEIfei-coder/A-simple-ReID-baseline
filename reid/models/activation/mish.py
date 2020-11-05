from __future__ import absolute_import

import torch
from torch.nn import functional as F
from torch import Tensor


def mish_(z: Tensor):
	'''
	:param z: inputs
	:return: z*tanh(ln(1+exp(z)))
	'''
	return z * (torch.tanh(F.softplus(z)))
