"""Attention-guided skip fusion, adapted from Oktay et al.'s official code.

Pinned reference: eee4881fdc31920efd873773e0b744df8dacbfb6.
Generic 2D gate includes W; GenCom adds the CT-Single combine block.
Only modernize APIs and make align_corners=False explicit inside the gate.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


def validate_attention(settings):
    if settings is None:
        return None
    if not isinstance(settings,dict):
        raise ValueError('attention must be a dictionary or None')
    expected={'strategy':settings.get('strategy'), 'block':settings.get('block'),
              'inter_ratio':settings.get('inter_ratio'), 'sub_sample_factor':[2,2],
              'align_corners':False, 'mode':'concatenation', 'gating_channels':128}
    if (expected['strategy'] not in ('PRG','FBG') or expected['block'] not in ('GenStan','GenCom')
            or expected['inter_ratio'] not in (0.5,1.0) or settings!=expected):
        raise ValueError('ATG NOW settings are locked; unknown or changed configuration')
    return deepcopy(expected)


def official_init(module):
    """Official Kaiming helper: leave Conv biases at constructor defaults."""
    for child in module.modules():
        if isinstance(child,nn.Conv2d):
            nn.init.kaiming_normal_(child.weight,a=0,mode='fan_in')
        elif isinstance(child,nn.BatchNorm2d):
            nn.init.normal_(child.weight,1.0,0.02)
            nn.init.zeros_(child.bias)


class GridAttentionBlock2D(nn.Module):
    def __init__(self,in_channels,gating_channels,inter_channels):
        super().__init__()
        self.W=nn.Sequential(nn.Conv2d(in_channels,in_channels,1),nn.BatchNorm2d(in_channels))
        self.theta=nn.Conv2d(in_channels,inter_channels,2,stride=2,padding=0,bias=False)
        self.phi=nn.Conv2d(gating_channels,inter_channels,1,bias=True)
        self.psi=nn.Conv2d(inter_channels,1,1,bias=True)
        # Same traversal/order as Generic author's gate; no global host re-init.
        for child in self.children():
            official_init(child)

    def forward(self,x,g):
        if x.shape[0]!=g.shape[0]:
            raise ValueError('Skip and gating batches must match')
        theta=self.theta(x)
        phi=F.interpolate(self.phi(g),size=theta.shape[-2:],mode='bilinear',align_corners=False)
        score=self.psi(F.relu(theta+phi,inplace=True))
        alpha=F.interpolate(torch.sigmoid(score),size=x.shape[-2:],mode='bilinear',align_corners=False)
        return self.W(alpha*x),alpha


class AttentionSkip(nn.Module):
    def __init__(self,channels,gating_channels,inter_channels,combine=False):
        super().__init__()
        self.gate=GridAttentionBlock2D(channels,gating_channels,inter_channels)
        self.combine_gates=None
        if combine:
            self.combine_gates=nn.Sequential(nn.Conv2d(channels,channels,1),
                                             nn.BatchNorm2d(channels),nn.ReLU(inplace=True))
            official_init(self.combine_gates)

    def forward(self,x,g):
        gated,alpha=self.gate(x,g)
        if self.combine_gates is not None:
            gated=self.combine_gates(gated)
        return gated,alpha


def make_attention(settings):
    settings=validate_attention(settings)
    gating=nn.Sequential(nn.Conv2d(128,128,1),nn.BatchNorm2d(128),nn.ReLU(inplace=True))
    official_init(gating)
    gates=nn.ModuleDict()
    for level,channels in [(4,128),(3,64),(2,32)]:
        cg=128 if settings['strategy']=='FBG' or level==4 else channels
        gates[str(level)]=AttentionSkip(channels,cg,int(channels*settings['inter_ratio']),
                                         combine=settings['block']=='GenCom')
    return gating,gates
