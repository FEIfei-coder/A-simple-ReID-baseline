from __future__ import absolute_import

import torch
import pytorch_metric_learning.losses as losses
import torch.nn as nn

from IPython import embed



class TriHardLoss(nn.Module):
	"""
	we achieve this loss which is from: https://arxiv.org/abs/1703.07737
    """
	def __init__(self, margin=0.3):
		'''
		:param margin: margin of trihard  <-->   L = max{max(d_{a, p}) - min(d_{a, n}) + m, 0}
		'''
		super(TriHardLoss, self).__init__()
		self.margin = margin
		self.rankingLoss = nn.MarginRankingLoss(margin=margin)

	def forward(self, inputs, targets):
		'''
		:param inputs: feature matrix with shape (batch_size, feat_dim)
		:param targets: ground truth labels with shape (batch_size)
		:return: trihard loss
		'''
		n = inputs.size(0)

		dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
		dist = dist + dist.t()
		dist.addmm_(1, -2, inputs, inputs.t())
		dist.clamp(1e-12).sqrt() #clamp the value
		mask = targets.expand(n, n).eq(targets.expand(n, n).t())
		dist_ap, dist_an = [], []
		for i in range(n):
			dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
			dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
		dist_ap = torch.cat(dist_ap)
		dist_an = torch.cat(dist_an)
		y = torch.ones_like(dist_an)
		loss = self.rankingLoss(dist_an, dist_ap, y)
		return loss





