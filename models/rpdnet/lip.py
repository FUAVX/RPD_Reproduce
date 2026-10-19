"""P3 LIP-Now: standalone DenseNet-style G adapted to RepDWNet.

Reference: sebgao/LIP@6724ec63c1f1ebefa25d7269e29d17c4411a7941,
imagenet/lip_densenet.py. Team choices: B=C/4, modern IN gamma=1,
Kaiming fan_in, and explicit zero final Conv (paper / ResNet init principle).
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

REFERENCE_COMMIT = '6724ec63c1f1ebefa25d7269e29d17c4411a7941'


def lip2d(x, logits):
    """Official primitive specialized to k=2, stride=2, padding=0."""
    weight = logits.exp()
    return F.avg_pool2d(x * weight, 2, 2, 0) / F.avg_pool2d(weight, 2, 2, 0)


class SoftGate(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).mul(12.0)


class BottleneckLIP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        if type(channels) is not int or channels < 4 or channels % 4:
            raise ValueError('LIP-Now requires channels >= 4 divisible by 4 (B=C/4).')
        self.channels = channels
        self.width = channels // 4
        b = self.width
        def norm(c):
            return nn.InstanceNorm2d(c, eps=1e-5, affine=True, track_running_stats=False)
        self.logit = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channels, b, 1, bias=False)),
            ('bn1', norm(b)), ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(b, b, 3, padding=1, bias=False)),
            ('bn2', norm(b)), ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(b, channels, 1, bias=False)),
            ('bn3', norm(channels)), ('gate', SoftGate()),
        ]))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.logit:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(layer, nn.InstanceNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.logit.conv3.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 2:
            raise ValueError(f'Expected [N,{self.channels},H>=2,W>=2], got {tuple(x.shape)}')
        return lip2d(x, self.logit(x))
