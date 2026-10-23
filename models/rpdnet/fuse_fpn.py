"""FPN-inspired Fuse only: local Add or a lazy multi-scale pyramid decoder.

Reference: FAIR Detectron FPN.py at 04155a01a6ea68f22ac27c79a822066457941ece.
Team adaptations: bilinear/align_corners=True, width32, no P5 smoothing,
optional DW+PW refinement, final mean and a single semantic logits head.
"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

OUTPUT_LEVELS = {'P52': (5, 4, 3, 2), 'P42': (4, 2), 'P43': (4, 3)}


def validate_fusion(settings):
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError('fusion must be a dictionary or None')
    variant = settings.get('variant')
    if variant in ('add', 'projected_add'):
        expected = {'variant': variant}
    elif variant == 'fpn':
        expected = {'variant':'fpn', 'refinement':settings.get('refinement'),
                    'pyramid':settings.get('pyramid'), 'channels':32, 'align_corners':True}
        if expected['refinement'] not in ('vani','dpw') or expected['pyramid'] not in OUTPUT_LEVELS:
            raise ValueError('Unknown FPN refinement/pyramid')
    else:
        raise ValueError('Unknown fusion variant')
    if settings != expected:
        raise ValueError('Fuse NOW settings are locked; unexpected or changed settings')
    return deepcopy(expected)


def c2_xavier_fill(conv):
    """Official fvcore c2_xavier_fill: Caffe2 XavierFill != xavier_uniform_."""
    nn.init.kaiming_uniform_(conv.weight, a=1)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


def conv(in_channels, out_channels, kernel_size, groups=1, bias=True):
    return c2_xavier_fill(nn.Conv2d(in_channels, out_channels, kernel_size,
                                  padding=kernel_size//2, groups=groups, bias=bias))


class AddUp(nn.Module):
    """Keep Control's upsample/pad, Cmid and Cout; change fusion input width."""
    def __init__(self, channels, out_channels, projected, block, deploy, use_se, branches):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Processing/head retain original initialization; new projection alone uses C2 fill.
        self.conv = block(channels, out_channels, mid_channels=channels,
                          deploy=deploy, use_se=use_se, num_dw_branches=branches)
        self.lateral = conv(channels, channels, 1) if projected else nn.Identity()

    def forward(self, decoder, skip):
        decoder = self.up(decoder)
        dy, dx = skip.shape[-2]-decoder.shape[-2], skip.shape[-1]-decoder.shape[-1]
        decoder = F.pad(decoder, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(decoder + self.lateral(skip))


class PyramidDecoder(nn.Module):
    def __init__(self, settings):
        super().__init__()
        settings = validate_fusion(settings)
        self.levels = OUTPUT_LEVELS[settings['pyramid']]
        self.minimum = min(self.levels)
        channels = {5:128, 4:128, 3:64, 2:32}
        self.laterals = nn.ModuleDict({str(i):conv(channels[i],32,1)
                                      for i in range(5,self.minimum-1,-1)})
        def refine():
            if settings['refinement'] == 'vani':
                return conv(32,32,3)
            return nn.Sequential(conv(32,32,3,groups=32,bias=False),conv(32,32,1))
        self.refinements = nn.ModuleDict({str(i):refine() for i in self.levels if i != 5})
        self.head = conv(32,3,1)

    @staticmethod
    def resize(feature, size):
        return F.interpolate(feature, size=size, mode='bilinear', align_corners=True)

    def forward(self, features, input_size):
        target = features[2].shape[-2:]  # also for P43: no lateral2/M2 is constructed
        merged = self.laterals['5'](features[5])
        accumulator = self.resize(merged,target) if 5 in self.levels else None
        for level in range(4,self.minimum-1,-1):
            merged = self.laterals[str(level)](features[level]) + self.resize(merged,features[level].shape[-2:])
            # Feed unrefined M into the next top-down step; create only selected P.
            if level in self.levels:
                output = self.resize(self.refinements[str(level)](merged),target)
                accumulator = output if accumulator is None else accumulator + output
        mean = accumulator / len(self.levels)
        return self.resize(self.head(mean),input_size)
