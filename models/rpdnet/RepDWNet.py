import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rpdnet.RPD_Module import RPD
from models.rpdnet.RPDNet import DoubleConv, OutConv
from models.rpdnet.rep_dw import RepDW


class DoubleConv_down_repdw(nn.Module):
    """
    Building block cho Encoder/Decoder của RepDWNet.
    Gồm 2 stage (Conv1 và Conv2). Mỗi stage cấu trúc Depthwise Separable:
      - Lớp 1: RepDW (Depthwise Conv 3x3 đa nhánh chuẩn)
      - Lớp 2: RPD('cv', kernel_size=1) (Pointwise Conv 1x1 đa nhánh chuẩn)
    """
    def __init__(self, in_channels, out_channels, mid_channels=None,
                 deploy=False, use_se=False, num_dw_branches=4):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.Conv1 = self._make_stage(in_channels, mid_channels, deploy, use_se, num_dw_branches)
        self.Conv2 = self._make_stage(mid_channels, out_channels, deploy, use_se, num_dw_branches)

    def forward(self, input):
        x = self.Conv1(input)
        x = self.Conv2(x)
        return x

    def _make_stage(self, in_channels, out_channels, deploy, use_se, num_dw_branches):
        layers = []
        # Lớp 1 (Depthwise): RepDW thay thế RPD('pdc')
        layers.append(RepDW(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=3, stride=1, groups=in_channels,
                            bias=True, deploy=deploy, use_se=use_se,
                            num_dw_branches=num_dw_branches))
        # Lớp 2 (Pointwise): RPD('cv') giữ nguyên 100% của tác giả
        layers.append(RPD('cv', in_channels=in_channels, out_channels=out_channels,
                          kernel_size=1, stride=1, groups=1, bias=True,
                          deploy=deploy, use_se=use_se, convert=False,
                          num_conv_branches=4))
        return nn.Sequential(*layers)


class Down_repdw(nn.Sequential):
    def __init__(self, in_channels, out_channels, deploy=False, use_se=False, num_dw_branches=4):
        super().__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_down_repdw(in_channels, out_channels, deploy=deploy,
                                  use_se=use_se, num_dw_branches=num_dw_branches)
        )


class Up_repdw(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, deploy=False,
                 use_se=False, num_dw_branches=4):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_down_repdw(in_channels, out_channels, in_channels // 2,
                                              deploy=deploy, use_se=use_se,
                                              num_dw_branches=num_dw_branches)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_down_repdw(in_channels, out_channels, deploy=deploy,
                                              use_se=use_se,
                                              num_dw_branches=num_dw_branches)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class RepDWNet(nn.Module):
    """
    RepDWNet: U-Net kiến trúc Depthwise Separable tái tham số hóa (Structural Re-parameterization).
    Thay thế các toán tử vi sai PDC bằng khối RepDW chuẩn, loại bỏ quy trình convert 2 bước.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 16,
                 deploy: bool = False,
                 use_se: bool = False,
                 num_dw_branches: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.base_c = base_c

        # Stem: 2x RepConvbn 3x3 (giữ nguyên gốc)
        self.in_conv = DoubleConv(in_channels, base_c, deploy=deploy)

        # Encoder: 4 stages
        self.down1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.down2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.down3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        factor = 2 if bilinear else 1
        self.down4 = Down_repdw(base_c * 8, base_c * 16 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Decoder: 4 stages
        self.up1 = Up_repdw(base_c * 16, base_c * 8 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.up2 = Up_repdw(base_c * 8, base_c * 4 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.up3 = Up_repdw(base_c * 4, base_c * 2 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.up4 = Up_repdw(base_c * 2, base_c, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Head: 1x1 Conv
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
