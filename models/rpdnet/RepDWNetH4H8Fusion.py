"""Phase 3 parallel /4-/8 candidate, using the supplied B1 blocks."""

import torch.nn.functional as F
from torch import nn

from models.rpdnet.RepDWNet import DoubleConv_down_repdw
from models.rpdnet.RepDWNetH8 import RepDWNetH8


def projection(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                         nn.BatchNorm2d(out_channels))


class ResolutionExchange(nn.Module):
    """Simultaneous bidirectional updates from the same pre-fusion tensors."""

    def __init__(self, high_channels, low_channels):
        super().__init__()
        self.low_to_high = projection(low_channels, high_channels)
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(low_channels))
        self.activation = nn.ReLU(inplace=False)

    def forward(self, high, low):
        from_low = F.interpolate(self.low_to_high(low), size=high.shape[-2:],
                                 mode="bilinear", align_corners=True)
        from_high = self.high_to_low(high)
        # MaxPool rounds down while stride-2 convolution can round up on odd sizes.
        if from_high.shape[-2:] != low.shape[-2:]:
            from_high = F.interpolate(from_high, size=low.shape[-2:],
                                      mode="bilinear", align_corners=True)
        return self.activation(high + from_low), self.activation(low + from_high)


class RepDWNetH4H8Fusion(RepDWNetH8):
    """Two processed streams, two exchanges by default; no /16 feature."""

    def __init__(self, num_classes, in_channels=3, bilinear=True, base_c=16,
                 deploy=False, use_se=False, num_dw_branches=4,
                 high_channels=32, exchange_units=2, exchange_mode="repeated"):
        if high_channels < 1 or exchange_units < 1:
            raise ValueError("high_channels and exchange_units must be positive.")
        if exchange_mode not in ("repeated", "final_only"):
            raise ValueError("exchange_mode must be repeated or final_only.")
        super().__init__(num_classes, in_channels=in_channels, bilinear=bilinear,
                         base_c=base_c, deploy=deploy, use_se=use_se,
                         num_dw_branches=num_dw_branches)
        # The low processing units replace the last encoder block, so there are
        # no trainable parameters in an unused inherited down4 branch.
        self.down4 = nn.Identity()
        self.high_channels = high_channels
        self.exchange_units = exchange_units
        self.exchange_mode = exchange_mode
        self.high_projection = projection(base_c * 4, high_channels)
        block_options = dict(deploy=deploy, use_se=use_se,
                             num_dw_branches=num_dw_branches)
        self.high_blocks = nn.ModuleList([
            DoubleConv_down_repdw(high_channels, high_channels, **block_options)
            for _ in range(exchange_units)])
        self.low_blocks = nn.ModuleList([
            DoubleConv_down_repdw(base_c * 8, base_c * 8, **block_options)
            for _ in range(exchange_units)])
        self.exchanges = nn.ModuleDict({
            str(i): ResolutionExchange(high_channels, base_c * 8)
            for i in range(exchange_units)
            if exchange_mode == "repeated" or i == exchange_units - 1})
        self.high_skip_projection = projection(high_channels, base_c * 4)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        high, low = self.high_projection(x3), x4
        for i, (high_block, low_block) in enumerate(zip(self.high_blocks, self.low_blocks)):
            high_pre, low_pre = high_block(high), low_block(low)
            if str(i) in self.exchanges:
                high, low = self.exchanges[str(i)](high_pre, low_pre)
            else:
                high, low = high_pre, low_pre
        x = self.up1(low, x4)
        x = self.up2(x, self.high_skip_projection(high))
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
