"""Tri-3 MaxBlurPool for RepDWNet (AAA-Now).

Reference: Richard Zhang, ICML 2019; adobe/antialiased-cnns,
commit b27a34a26f3ab039113d44d83c54d0428598ac9c, blurpool.py / vgg.py.
This project adaptation uses standard PyTorch operations and fixed coefficients.
"""
import torch
from torch import nn
from torch.nn import functional as F

REFERENCE_COMMIT = 'b27a34a26f3ab039113d44d83c54d0428598ac9c'


def tri3_filter(channels, dtype=torch.float32, device=None):
    a = torch.tensor([1., 2., 1.], dtype=dtype, device=device)
    return (a[:, None] * a[None, :] / 16)[None, None].repeat(channels, 1, 1, 1)


class AAAMaxBlurPool(nn.Module):
    """Valid dense Max2, reflect-pad1, fixed Tri-3 grouped convolution/s2.

    Consumes no RNG and adds no Parameters. Spatial inputs must be >=3.
    Output shape is floor(H/2), floor(W/2), including odd inputs.
    """
    def __init__(self, channels, filt_size=3, pad_type='reflect', stride=2, pad_off=0):
        super().__init__()
        if type(channels) is not int or channels < 1:
            raise ValueError('AAA channels must be a positive integer.')
        if (type(filt_size) is not int or filt_size != 3 or pad_type != 'reflect'
                or type(stride) is not int or stride != 2 or type(pad_off) is not int or pad_off != 0):
            raise ValueError('AAA-Now requires filt_size=3, pad_type=reflect, stride=2, pad_off=0.')
        self.channels, self.filt_size = channels, filt_size
        self.pad_type, self.stride, self.pad_off = pad_type, stride, pad_off
        self.maxpool = nn.MaxPool2d(2, stride=1, padding=0, ceil_mode=False)
        self.pad = nn.ReflectionPad2d(1)
        self.register_buffer('filt', tri3_filter(channels))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels or min(x.shape[-2:]) < 3:
            raise ValueError(f'AAA expects [N,{self.channels},H>=3,W>=3]; got {tuple(x.shape)}')
        return F.conv2d(self.pad(self.maxpool(x)), self.filt, stride=2, groups=self.channels)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        tensor = state_dict.get(prefix+'filt')
        if isinstance(tensor, torch.Tensor):
            expected = tri3_filter(self.channels, tensor.dtype, tensor.device)
            if not torch.equal(tensor, expected):
                error_msgs.append(prefix+'filt must contain the exact fixed normalized Tri-3 kernel')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
