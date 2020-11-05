from __future__ import absolute_import

import torch.nn as nn
import torch
import torchvision.models
import torch.nn.functional as F

from IPython import embed


class ResNet50(nn.Module):
	'''
	we get the model resnet50 and handle it for what we want
	'''
	def __init__(self, num_classes, loss={'xent'},  **kwags):
		super(ResNet50, self).__init__()
		resnet = torchvision.models.resnet50(pretrained=False)
		self.base = nn.Sequential(*list(resnet.children())[:-2])
		self.classifier = nn.Linear(2048, num_classes)
		# self.training = training
		self.feature_dim = 2048

		self.loss = loss

	def forward(self, input):
		x = self.base(input)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size()[0], -1)

		if not self.training or self.loss == {'htri'}:
			return f
		y = self.classifier(f)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f
		elif self.loss == {'cent'}:
			return y, f
		elif self.loss == {'ring'}:
			return y, f
		else:
			raise KeyError("Unsupported loss_fuc: {}".format(self.loss))


