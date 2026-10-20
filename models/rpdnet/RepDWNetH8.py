"""Phase 3 /8 variant; retain baseline blocks and parameter names."""

from torch import nn

from models.rpdnet.RepDWNet import RepDWNet


class RepDWNetH8(RepDWNet):
    """Remove only the last encoder pool and first decoder upsample."""

    def __init__(self, num_classes, in_channels=3, bilinear=True, base_c=16,
                 deploy=False, use_se=False, num_dw_branches=4):
        if not bilinear:
            raise ValueError("RepDWNetH8 supports the baseline bilinear decoder only.")
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         bilinear=bilinear, base_c=base_c, deploy=deploy,
                         use_se=use_se, num_dw_branches=num_dw_branches)
        self.down4[0] = nn.Identity()
        self.up1.up = nn.Identity()
