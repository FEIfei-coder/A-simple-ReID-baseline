from __future__ import absolute_import

import torch
from torch import Tensor
import torch.nn.functional as F


def cosine_(x: Tensor, y: Tensor, dim: int =1):
	return F.cosine_similarity(x, y, dim)


