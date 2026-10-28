"""U-Net++-inspired nested fusion adapted to P2 RepDWNet (team implementation).

Reference: MrGiovanni/UNetPlusPlus e9153d4243e7bb67880c380ce803b176c81476f1,
Keras standard_unit; initializer semantics from Keras2.2.4.
Changes: PyTorch/NCHW, P2 bilinear x2+pad, RepDW diagonal nodes, fixed mids,
raw3-class head, P2 coupled Adam WD rather than an extra kernel-L2 objective.
Source attribution/license and modification notes: tests/reference/ndc/NOTICE.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

SETTINGS={'variant':'full','h_dropout':0.5,'h_init':'keras224_truncated_he',
          'decoder_mid_channels':[128,64,32,16],'align_corners':True,
          'resize':'bilinear_x2_pad','concat_order':'upsampled_encoder_previous',
          'extra_kernel_l2':False}
KERAS_TRUNCATED_NORMAL_STD=0.87962566103423978


def validate_nested(settings):
    if settings is None:
        return None
    if not isinstance(settings,dict) or settings!=SETTINGS:
        raise ValueError('NDC NOW settings are locked to the approved full topology/policies')
    return deepcopy(SETTINGS)


def keras_he_normal(conv):
    """Keras2.2.4 He normal distribution: truncated at +/-2 pre-truncation std."""
    fan_in=conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1]
    std_pre=math.sqrt(2.0/fan_in)/KERAS_TRUNCATED_NORMAL_STD
    nn.init.trunc_normal_(conv.weight,mean=0.0,std=std_pre,a=-2*std_pre,b=2*std_pre)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class HUnit(nn.Sequential):
    """Two independent standard convolutions/ReLU/element-wise Dropout, no norm."""
    def __init__(self,in_channels,out_channels):
        super().__init__(
            nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1,bias=True),
            nn.ReLU(),nn.Dropout(p=0.5),
            nn.Conv2d(out_channels,out_channels,3,stride=1,padding=1,bias=True),
            nn.ReLU(),nn.Dropout(p=0.5))
        keras_he_normal(self[0]);keras_he_normal(self[3])


def upsample_like(deeper,target):
    """Exact P2 resize semantics, including odd-size right/bottom padding."""
    up=F.interpolate(deeper,scale_factor=2,mode='bilinear',align_corners=True)
    dy,dx=target.shape[-2]-up.shape[-2],target.shape[-1]-up.shape[-1]
    return F.pad(up,[dx//2,dx-dx//2,dy//2,dy-dy//2])


def merge_nested(deeper,previous):
    """One Concat only; ordered upsampled, encoder, previous same-level H nodes."""
    if not previous:
        raise ValueError('Nested fusion requires a target-level encoder feature')
    return torch.cat([upsample_like(deeper,previous[0]),*previous],dim=1)
