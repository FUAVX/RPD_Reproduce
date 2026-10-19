"""
Detail Spatial Path (DSP) Modules and RepDWNet_DSP Architecture for P3_DefiRPD.
Implements 4 ablation candidates:
  - DSP-Ex1: Canonical DW-Spatial Path (Single-tap H/4 Concat)
  - DSP-Ex2: Rep-Structural Detail Path (Multi-branch RepDW training, single-kernel deploy)
  - DSP-Ex3: BiSe-FFM Attention Gating (Feature Fusion Module with Channel Attention)
  - DSP-Ex4: PID-Bilateral Dual-Scale (Two-tap H/2 and H/4 injection)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rpdnet.RPDNet import DoubleConv, OutConv
from models.rpdnet.RepDWNet import Down_repdw, Up_repdw, DoubleConv_down_repdw
from models.rpdnet.rep_dw import RepDW


class BiSeFFM_Block(nn.Module):
    """
    BiSeNet Feature Fusion Module (FFM) with Channel Attention Gating.
    Fuses low-level high-resolution detail features and high-level semantic context features,
    dynamically filtering out background soil noise while highlighting weed plant textures.
    """
    def __init__(self, in_channels_ctx: int, in_channels_detail: int, out_channels: int, reduction: int = 4):
        super().__init__()
        in_channels = in_channels_ctx + in_channels_detail
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        mid_channels = max(8, out_channels // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_attn = nn.Sequential(
            nn.Conv2d(out_channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_ctx: torch.Tensor, x_detail: torch.Tensor) -> torch.Tensor:
        # 1. Concat context and detail along channel dimension
        x_cat = torch.cat([x_ctx, x_detail], dim=1)
        # 2. Linear projection to unified feature space
        f_proj = self.conv_proj(x_cat)
        # 3. Squeeze & Excitation channel gating
        w = self.channel_attn(self.gap(f_proj))
        # 4. Residual channel-wise scaling
        return f_proj * (1.0 + w)


class DetailBlock_DW(nn.Module):
    """
    Lightweight Depthwise Separable Block for Detail Spatial Path.
    Either uses standard DW+PW or structural multi-branch RepDW.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 use_repdws: bool = False, deploy: bool = False, num_dw_branches: int = 4):
        super().__init__()
        self.use_repdws = use_repdws
        if use_repdws:
            self.dw = RepDW(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=3, stride=stride, groups=in_channels,
                            bias=True, deploy=deploy, use_se=False,
                            num_dw_branches=num_dw_branches)
        else:
            self.dw = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                          padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        self.pw = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DetailSpatialPath(nn.Module):
    """
    Detail Spatial Path (DSP) for Preserving High-Resolution Geometry.
    Outputs:
      - Single-tap: F_H4 (H/4, 256x256, 32 channels)
      - Dual-tap: (F_H2, F_H4) where F_H2 is (H/2, 512x512, 16 channels)
    """
    def __init__(self, in_channels: int = 3, base_c: int = 16, out_c: int = 32,
                 dual_scale: bool = False, use_repdws: bool = False,
                 deploy: bool = False, num_dw_branches: int = 4):
        super().__init__()
        self.dual_scale = dual_scale

        # Stage 1: Stem downsampling H -> H/2 (1024 -> 512)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True)
        )

        # Stage 2: Downsampling H/2 -> H/4 (512 -> 256)
        self.block1 = DetailBlock_DW(base_c, out_c, stride=2,
                                     use_repdws=use_repdws, deploy=deploy,
                                     num_dw_branches=num_dw_branches)

        # Stage 3: Refinement at H/4 (256 -> 256)
        self.block2 = DetailBlock_DW(out_c, out_c, stride=1,
                                     use_repdws=use_repdws, deploy=deploy,
                                     num_dw_branches=num_dw_branches)

    def forward(self, x: torch.Tensor):
        f_h2 = self.stem(x)       # [B, 16, 512, 512]
        f_h4 = self.block1(f_h2)   # [B, 32, 256, 256]
        f_h4 = self.block2(f_h4)   # [B, 32, 256, 256]

        if self.dual_scale:
            return f_h2, f_h4
        return f_h4


class RepDWNet_DSP(nn.Module):
    """
    RepDWNet with Detail Spatial Path (Two-Stream Architecture).
    Supports 4 experimental variants:
      - 'dsp_ex1_lite': Standard DWConv DSP with single-tap H/4 concat injection at up2.
      - 'dsp_ex2_rep': RepDW multi-branch DSP with single-tap H/4 concat injection at up2.
      - 'dsp_ex3_ffm': DWConv DSP with BiSe-FFM Channel Attention Gating at up2.
      - 'dsp_ex4_dual': Dual-scale DSP injecting H/2 at up3 and H/4 at up2.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 16,
                 deploy: bool = False,
                 use_se: bool = False,
                 num_dw_branches: int = 4,
                 dsp_variant: str = 'dsp_ex1_lite'):
        super().__init__()
        valid_variants = ['dsp_ex1_lite', 'dsp_ex2_rep', 'dsp_ex3_ffm', 'dsp_ex4_dual']
        assert dsp_variant in valid_variants, f"Invalid dsp_variant: {dsp_variant}. Expected one of {valid_variants}"

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.base_c = base_c
        self.deploy = deploy
        self.dsp_variant = dsp_variant

        use_repdws = (dsp_variant == 'dsp_ex2_rep')
        self.use_ffm = (dsp_variant == 'dsp_ex3_ffm')
        self.dual_scale = (dsp_variant == 'dsp_ex4_dual')

        factor = 2 if bilinear else 1

        # ==================== 1. DETAIL SPATIAL PATH ====================
        self.dsp = DetailSpatialPath(
            in_channels=in_channels,
            base_c=base_c,
            out_c=base_c * 2,  # 32 channels
            dual_scale=self.dual_scale,
            use_repdws=use_repdws,
            deploy=deploy,
            num_dw_branches=num_dw_branches
        )

        # ==================== 2. MAIN ENCODER (CONTEXT PATH) ====================
        # Stem: 2x RepConvbn 3x3 (Output x1: H x W, 16ch)
        self.in_conv = DoubleConv(in_channels, base_c, deploy=deploy)

        # Encoder stages
        self.down1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)       # H/2, 32ch
        self.down2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # H/4, 64ch
        self.down3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # H/8, 128ch
        self.down4 = Down_repdw(base_c * 8, base_c * 16 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches) # H/16, 128ch

        # ==================== 3. DECODER ====================
        # Decoder Stage 1: up1 (H/16 -> H/8, 64ch)
        self.up1 = Up_repdw(base_c * 16, base_c * 8 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Decoder Stage 2: up2 (H/8 -> H/4, 32ch)
        self.up2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(base_c * 8 // factor, base_c * 8 // factor // 2, kernel_size=2, stride=2)
        
        if self.use_ffm:
            # Context = up(up1) (64) + skip x3 (64) = 128ch; Detail = 32ch; FFM projects to 64ch
            self.ffm_up2 = BiSeFFM_Block(in_channels_ctx=base_c * 8, in_channels_detail=base_c * 2, out_channels=base_c * 4)
            self.up2_conv = DoubleConv_down_repdw(base_c * 4, base_c * 4 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        else:
            # Concat: up(up1) (64) + skip x3 (64) + detail (32) = 160ch -> 32ch
            self.up2_conv = DoubleConv_down_repdw(base_c * 8 + base_c * 2, base_c * 4 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Decoder Stage 3: up3 (H/4 -> H/2, 16ch)
        if self.dual_scale:
            self.up3_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(base_c * 4 // factor, base_c * 4 // factor // 2, kernel_size=2, stride=2)
            # Concat: up(up2) (32) + skip x2 (32) + detail_h2 (16) = 80ch -> 16ch
            self.up3_conv = DoubleConv_down_repdw(base_c * 4 + base_c, base_c * 2 // factor, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        else:
            self.up3 = Up_repdw(base_c * 4, base_c * 2 // factor, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Decoder Stage 4: up4 (H/2 -> H, 16ch)
        self.up4 = Up_repdw(base_c * 2, base_c, bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # ==================== 4. HEAD ====================
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 1. Detail Spatial Path ---
        if self.dual_scale:
            f_detail_h2, f_detail_h4 = self.dsp(x)
        else:
            f_detail_h4 = self.dsp(x)

        # --- 2. Main Encoder Path ---
        x1 = self.in_conv(x)      # [B, 16, 1024, 1024]
        x2 = self.down1(x1)       # [B, 32, 512, 512]
        x3 = self.down2(x2)       # [B, 64, 256, 256]
        x4 = self.down3(x3)       # [B, 128, 128, 128]
        x5 = self.down4(x4)       # [B, 128, 64, 64]

        # --- 3. Decoder with DSP Fusion ---
        # Stage 1: up1
        x_up1 = self.up1(x5, x4)  # [B, 64, 128, 128]

        # Stage 2: up2 with H/4 Detail Injection
        x_up1_256 = self.up2_upsample(x_up1)
        # Pad if odd dimensions
        diff_y = x3.size()[2] - x_up1_256.size()[2]
        diff_x = x3.size()[3] - x_up1_256.size()[3]
        if diff_x != 0 or diff_y != 0:
            x_up1_256 = F.pad(x_up1_256, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        if self.use_ffm:
            x_ctx = torch.cat([x_up1_256, x3], dim=1)  # [B, 128, 256, 256]
            f_fused = self.ffm_up2(x_ctx, f_detail_h4) # [B, 64, 256, 256]
            x_up2 = self.up2_conv(f_fused)             # [B, 32, 256, 256]
        else:
            x_in_up2 = torch.cat([x_up1_256, x3, f_detail_h4], dim=1) # [B, 160, 256, 256]
            x_up2 = self.up2_conv(x_in_up2)                           # [B, 32, 256, 256]

        # Stage 3: up3 (with H/2 Detail Injection if Ex4)
        if self.dual_scale:
            x_up2_512 = self.up3_upsample(x_up2)
            diff_y = x2.size()[2] - x_up2_512.size()[2]
            diff_x = x2.size()[3] - x_up2_512.size()[3]
            if diff_x != 0 or diff_y != 0:
                x_up2_512 = F.pad(x_up2_512, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x_in_up3 = torch.cat([x_up2_512, x2, f_detail_h2], dim=1) # [B, 80, 512, 512]
            x_up3 = self.up3_conv(x_in_up3)                           # [B, 16, 512, 512]
        else:
            x_up3 = self.up3(x_up2, x2)                               # [B, 16, 512, 512]

        # Stage 4: up4
        x_up4 = self.up4(x_up3, x1)                                   # [B, 16, 1024, 1024]

        # --- 4. Segmentation Head ---
        logits = self.out_conv(x_up4)                                 # [B, 3, 1024, 1024]
        return logits
