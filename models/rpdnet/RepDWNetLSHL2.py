"""PhuongTU — L2: mean decoder logits at /1, /2 and /4 on unchanged B1."""
from torch import nn
import torch.nn.functional as F

from .RepDWNetLSHL1 import RepDWNetLSHL1


class RepDWNetLSHL2(RepDWNetLSHL1):
    def __init__(self, num_classes, in_channels=3, bilinear=True, base_c=16,
                 deploy=False, use_se=False, num_dw_branches=4):
        super().__init__(num_classes, in_channels=in_channels, bilinear=bilinear,
                         base_c=base_c, deploy=deploy, use_se=use_se,
                         num_dw_branches=num_dw_branches)
        self.head_s4 = nn.Conv2d(base_c * 2, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        f1, f2, f4 = self._decoder_features(x)
        z1 = self.out_conv(f1)
        z2 = F.interpolate(self.head_s2(f2), size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)
        z4 = F.interpolate(self.head_s4(f4), size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)
        return (z1 + z2 + z4) / 3
