import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rpdnet.RPD_Module import RPD
from models.rpdnet.RPDNet import DoubleConv, OutConv
from models.rpdnet.rep_dw import RepDW
from models.rpdnet.fuse_ndc import validate_nested, HUnit, merge_nested


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
                 use_se=False, num_dw_branches=4, mid_channels=None):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_down_repdw(in_channels, out_channels,
                                              in_channels // 2 if mid_channels is None else mid_channels,
                                              deploy=deploy, use_se=use_se,
                                              num_dw_branches=num_dw_branches)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_down_repdw(in_channels, out_channels, deploy=deploy,
                                              use_se=use_se,
                                              num_dw_branches=num_dw_branches)

    def forward(self, x1, x2):
        skips = list(x2) if isinstance(x2, (list, tuple)) else [x2]
        if not skips:
            raise ValueError('Up requires at least one same-level feature')
        x1 = self.up(x1)
        diff_y = skips[0].size()[2] - x1.size()[2]
        diff_x = skips[0].size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x1, *skips], dim=1)
        return self.conv(x)


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
                 num_dw_branches: int = 4,
                 nested=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.base_c = base_c
        self.nested = validate_nested(nested)
        if self.nested is not None and (base_c != 16 or not bilinear or deploy or num_classes != 3):
            raise ValueError("NDC NOW requires base_c16, bilinear, 3 classes and unfused graph")

        # Stem: 2x RepConvbn 3x3 (giữ nguyên gốc)
        self.in_conv = DoubleConv(in_channels, base_c, deploy=deploy)

        # Encoder: 4 stages
        self.down1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.down2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.down3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        factor = 2 if bilinear else 1
        self.down4 = Down_repdw(base_c * 8, base_c * 16 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Decoder: 4 stages
        self.up1 = Up_repdw(base_c * 16, base_c * 8 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches,
                            mid_channels=128 if self.nested is not None else None)
        self.up2 = Up_repdw(base_c * 8, base_c * 4 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.up3 = Up_repdw(base_c * 4, base_c * 2 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.up4 = Up_repdw(base_c * 2, base_c, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Head: 1x1 Conv
        self.out_conv = OutConv(base_c, num_classes)
        if self.nested is not None:
            # Preserve P2 encoder, Up1 and head initialization and following RNG.
            with torch.random.fork_rng(devices=[]):
                self.up2 = Up_repdw(192,32,True,deploy,use_se,num_dw_branches,mid_channels=64)
                self.up3 = Up_repdw(128,16,True,deploy,use_se,num_dw_branches,mid_channels=32)
                self.up4 = Up_repdw(80,16,True,deploy,use_se,num_dw_branches,mid_channels=16)
                self.h_units = nn.ModuleDict({
                    name: HUnit(cin,cout) for name,cin,cout in [
                        ('H01',48,16),('H02',64,16),('H03',80,16),
                        ('H11',96,32),('H12',128,32),('H21',192,64)]})


    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.nested is not None:
            h01 = self.h_units['H01'](merge_nested(x2,[x1]))
            h11 = self.h_units['H11'](merge_nested(x3,[x2]))
            h21 = self.h_units['H21'](merge_nested(x4,[x3]))
            d4 = self.up1(x5,[x4])
            h02 = self.h_units['H02'](merge_nested(h11,[x1,h01]))
            h12 = self.h_units['H12'](merge_nested(h21,[x2,h11]))
            d3 = self.up2(d4,[x3,h21])
            h03 = self.h_units['H03'](merge_nested(h12,[x1,h01,h02]))
            d2 = self.up3(d3,[x2,h11,h12])
            d1 = self.up4(d2,[x1,h01,h02,h03])
            return self.out_conv(d1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
