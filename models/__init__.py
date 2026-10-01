from typing import Dict
from torch import nn

from .deeplabv3plus.modeling import deeplabv3plus_resnet50
from .rpdnet import RPDNet, RepDWNet
from .erf import ERFNetModel
from .segnext import SegNext
from .segformer import SegFormer
from .efficientvit import EfficientViTSeg
from .model_multimetrics import *
from . import model_multimetrics as module
from .fgnet import FGNet
from .losses import *
from .rpdnet.RPD_ops import *


def get_backbone(cfg: Dict) -> nn.Module:
    num_classes = cfg['backbone']['num_classes']
    pretrained = cfg['backbone']['pretrained']

    if cfg['backbone']['name'] == 'deeplabv3plus_resnet50':
        return deeplabv3plus_resnet50(output_stride=16, num_classes=num_classes, pretrained_backbone=pretrained)

    if cfg['backbone']['name'] == 'erfnet':
        return ERFNetModel(num_classes, pretrained=pretrained)

    if cfg['backbone']['name'] == 'segnext':
        return SegNext(num_classes)

    if cfg['backbone']['name'] == 'segformer':
        return SegFormer(num_classes)

    if cfg['backbone']['name'] == 'efficientvitseg':
        return EfficientViTSeg(num_classes)

    if cfg['backbone']['name'] == 'fgnet':
        return FGNet(num_classes)

    if cfg['backbone']['name'] == 'RPDNet':
        deploy = cfg['backbone'].get('deploy', False)
        convert = cfg['backbone'].get('convert', False)
        return RPDNet(num_classes, deploy=deploy, convert=convert)

    if cfg['backbone']['name'] == 'RepDWNet':
        deploy = cfg['backbone'].get('deploy', False)
        use_se = cfg['backbone'].get('use_se', False)
        num_dw_branches = cfg['backbone'].get('num_dw_branches', 4)
        base_c = cfg['backbone'].get('base_c', 16)
        return RepDWNet(num_classes, deploy=deploy, use_se=use_se,
                        num_dw_branches=num_dw_branches, base_c=base_c)

    raise ValueError('The requested backbone is not supported.')







