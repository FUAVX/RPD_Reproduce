"""Downsampling selection without changing the existing MaxPool/DPP recipes."""
from .dpp import make_downsampling as make_dpp_downsampling
from .lip import BottleneckLIP
from .aaa import AAAMaxBlurPool
import torch
from torch import nn


def make_downsampling(channels, config=None, stage=None):
    if config is not None and config.get('variant') == 'aaa_tri3':
        allowed = {'variant', 'stages', 'filt_size', 'pad_type', 'stride', 'pad_off'}
        if set(config) - allowed:
            raise ValueError(f'Unknown AAA settings: {sorted(set(config)-allowed)}')
        stages = config.get('stages')
        if (not isinstance(stages, (list, tuple)) or any(type(s) is not int for s in stages)
                or list(stages) not in ([1, 2], [3, 4], [1, 2, 3, 4])):
            raise ValueError('AAA-Now placements: Early [1,2], Late [3,4], All [1,2,3,4].')
        # Require explicit geometry even at stages that keep MaxPool.
        expected = {'filt_size': 3, 'pad_type': 'reflect', 'stride': 2, 'pad_off': 0}
        if any(config.get(k) != v or type(config.get(k)) is not type(v) for k, v in expected.items()):
            raise ValueError(f'AAA requires explicit {expected}')
        if stage not in stages:
            return nn.MaxPool2d(2, stride=2)
        return AAAMaxBlurPool(channels, **{k: config[k] for k in expected})
    if config is None or config.get('variant', 'maxpool') != 'lip_bottleneck':
        return make_dpp_downsampling(channels, config, stage)
    unknown = set(config) - {'variant', 'stages'}
    if unknown:
        raise ValueError(f'Unknown LIP-Now settings: {sorted(unknown)}')
    stages = config.get('stages', [])
    if (not isinstance(stages, (list, tuple)) or len(stages) != 2
            or any(type(s) is not int or s not in (1, 2, 3, 4) for s in stages)
            or len(set(stages)) != 2):
        raise ValueError('LIP-Now requires exactly two distinct stages from 1..4.')
    if list(stages) not in ([1, 2], [3, 4], [2, 3]):
        raise ValueError('LIP-Now supports Early [1,2], Late [3,4], Middle [2,3].')
    if stage not in stages:
        return nn.MaxPool2d(2, stride=2)
    # get_backbone constructs the network on CPU before moving it to its device.
    # Initialize G from the current seed stream, but restore that stream before
    # constructing the next backbone layer. No extra seed or initializer is used.
    with torch.random.fork_rng(devices=[]):
        return BottleneckLIP(channels)
