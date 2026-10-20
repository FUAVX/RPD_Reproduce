"""PhuongTU — L1: mean decoder logits at /1 and /2 on unchanged B1."""
from torch import nn
import torch.nn.functional as F

from .RepDWNet import RepDWNet


class RepDWNetLSHL1(RepDWNet):
    def __init__(self, num_classes, in_channels=3, bilinear=True, base_c=16,
                 deploy=False, use_se=False, num_dw_branches=4):
        if not bilinear:
            raise ValueError("LSH ablations require the baseline bilinear decoder.")
        super().__init__(num_classes, in_channels=in_channels, bilinear=bilinear,
                         base_c=base_c, deploy=deploy, use_se=use_se,
                         num_dw_branches=num_dw_branches)
        # Retain and use out_conv as the /1 classifier, including its bias.
        self.head_s2 = nn.Conv2d(base_c, num_classes, kernel_size=1, bias=True)

    def _decoder_features(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d8 = self.up1(x5, x4)
        f4 = self.up2(d8, x3)
        f2 = self.up3(f4, x2)
        f1 = self.up4(f2, x1)
        return f1, f2, f4

    def forward(self, x):
        f1, f2, _ = self._decoder_features(x)
        z1 = self.out_conv(f1)
        z2 = F.interpolate(self.head_s2(f2), size=f1.shape[-2:],
                           mode="bilinear", align_corners=False)
        return (z1 + z2) / 2
