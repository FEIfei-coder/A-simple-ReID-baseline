from __future__ import absolute_import

import torch.nn as nn
import torch
import torchvision.models
import torch.nn.functional as F


class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(pretrained=True)
        self.base = nn.Sequential(*list(resnet101.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
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
