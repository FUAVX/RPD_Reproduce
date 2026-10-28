"""ICNet-inspired CFF, adapted to P2 resize and decoder widths by the team.

Reference: hszhao/ICNet dd2bf78ff3f9ff05f6d35c40a3ee216158d9e892,
evaluation/prototxt/icnet_cityscapes.prototxt and paper section3.3.
New PyTorch implementation; not copied Caffe source. Training BN/bias/init
are team policies, not a claim of exact author training-source reproduction.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

POLICIES = {'variant':'cff','decoder_mid_channels':[128,64,32,16],
            'align_corners':True,'resize':'bilinear_x2_pad','conv_bias':False,
            'conv_init':'kaiming_normal_fan_in_a0','bn_eps':1e-5,'bn_momentum':0.1,
            'bn_affine':True,'bn_track_running_stats':True,'bn_gamma':1.0,'bn_beta':0.0}

def settings(dilation=2,skip_mode='projection_bn'):
    return dict(deepcopy(POLICIES),dilation=dilation,skip_mode=skip_mode)

def validate_cff(value):
    if value is None:return None
    allowed=[settings(d,'projection_bn') for d in (1,2,3)]+[settings(2,s) for s in ('bn_only','identity')]
    if not isinstance(value,dict) or value not in allowed:
        raise ValueError('CFF NOW settings must match one of the five approved experiments')
    return deepcopy(value)

def upsample_like(deeper,skip):
    up=F.interpolate(deeper,scale_factor=2,mode='bilinear',align_corners=True)
    dy,dx=skip.shape[-2]-up.shape[-2],skip.shape[-1]-up.shape[-1]
    return F.pad(up,[dx//2,dx-dx//2,dy//2,dy-dy//2])

class CascadeFeatureFusion(nn.Module):
    def __init__(self,channels,cfg):
        super().__init__();self.settings=validate_cff(cfg)
        def bn():return nn.BatchNorm2d(channels,eps=cfg['bn_eps'],momentum=cfg['bn_momentum'],affine=True,track_running_stats=True)
        # The new CFF must not shift downstream RepDW/common-host initialization
        # when a skip projection is removed in an ablation.
        with torch.random.fork_rng(devices=[]):
            d=cfg['dilation']
            self.deep=nn.Sequential(nn.Conv2d(channels,channels,3,padding=d,dilation=d,bias=False),bn())
            nn.init.kaiming_normal_(self.deep[0].weight,a=0,mode='fan_in',nonlinearity='relu')
            mode=cfg['skip_mode']
            if mode=='projection_bn':
                self.skip=nn.Sequential(nn.Conv2d(channels,channels,1,bias=False),bn())
                nn.init.kaiming_normal_(self.skip[0].weight,a=0,mode='fan_in',nonlinearity='relu')
            elif mode=='bn_only':self.skip=bn()
            else:self.skip=nn.Identity()
            for layer in self.modules():
                if isinstance(layer,nn.BatchNorm2d):nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
        self.relu=nn.ReLU()
    def forward(self,deeper,skip):
        return self.relu(self.deep(upsample_like(deeper,skip))+self.skip(skip))

class CFFUp(nn.Module):
    def __init__(self,channels,out_channels,cfg,block,deploy,use_se,branches):
        super().__init__()
        self.conv=block(channels,out_channels,mid_channels=channels,deploy=deploy,use_se=use_se,num_dw_branches=branches)
        self.fuse=CascadeFeatureFusion(channels,cfg)
    def forward(self,deeper,skip):return self.conv(self.fuse(deeper,skip))
