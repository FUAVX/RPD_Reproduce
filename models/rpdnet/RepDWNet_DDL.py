import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rpdnet.RPDNet import DoubleConv, OutConv
from models.rpdnet.RepDWNet import DoubleConv_down_repdw, Down_repdw, Up_repdw


class RepDWNet_DDL(nn.Module):
    """
    RepDWNet_DDL (Dropped-Down Layers):
    Biến thể rút gọn cấu trúc từ 4-stage U-Net xuống 3-stage U-Net (3 Encoder + 3 Decoder blocks).
    Hỗ trợ 4 cấu hình loại bỏ tương ứng 1 cặp Encoder-Decoder (drop_pair in [1, 2, 3, 4]):
      - drop_pair=1 (DDL-Ex1): Bỏ Pair 1 (Down1, Up4) - Bỏ tầng chi tiết nông nhất (H <-> H/2).
      - drop_pair=2 (DDL-Ex2): Bỏ Pair 2 (Down2, Up3) - Bỏ tầng trung gian nông (H/2 <-> H/4).
      - drop_pair=3 (DDL-Ex3): Bỏ Pair 3 (Down3, Up2) - Bỏ tầng trung gian sâu (H/4 <-> H/8).
      - drop_pair=4 (DDL-Ex4): Bỏ Pair 4 (Down4, Up1) - Bỏ tầng nén sâu nhất (H/8 <-> H/16), giữ trần H/8.
    
    Cả 4 cấu hình đều:
      - Có Bottleneck tại độ phân giải H/8.
      - Phục hồi chính xác 100% về kích thước ảnh gốc H x W.
      - Hỗ trợ đầy đủ Structural Re-parameterization (switch_to_deploy) với deploy parity error < 1e-4.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 16,
                 drop_pair: int = 4,
                 deploy: bool = False,
                 use_se: bool = False,
                 num_dw_branches: int = 4):
        super().__init__()
        assert drop_pair in [1, 2, 3, 4], f"drop_pair must be in [1, 2, 3, 4], got {drop_pair}"

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.bilinear = bilinear
        self.base_c = base_c
        self.drop_pair = drop_pair
        self.deploy = deploy

        # Stem: 2x RepConvbn 3x3 (giữ nguyên gốc)
        self.in_conv = DoubleConv(in_channels, base_c, deploy=deploy)

        # -------------------------------------------------------------
        # Xây dựng 3 block Encoder và 3 block Decoder tùy theo drop_pair
        # -------------------------------------------------------------
        if drop_pair == 4:
            # DDL-Ex4: Bỏ Pair 4 (Down4, Up1). Giữ Down1, Down2, Down3; Up2, Up3, Up4.
            # Encoder
            self.enc1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)       # 16 -> 32, H/2
            self.enc2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 32 -> 64, H/4
            self.enc3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 64 -> 128, H/8 (Bottleneck)
            # Decoder
            # dec1: input x3 (128) upsample H/8 -> H/4, concat skip x2 (64) => in_c = 128 + 64 = 192, out_c = 32
            self.dec1 = Up_repdw(base_c * 12, base_c * 2, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec2: input (32) upsample H/4 -> H/2, concat skip x1 (32) => in_c = 32 + 32 = 64, out_c = 16
            self.dec2 = Up_repdw(base_c * 4, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec3: input (16) upsample H/2 -> H, concat skip x0 (16) => in_c = 16 + 16 = 32, out_c = 16
            self.dec3 = Up_repdw(base_c * 2, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        elif drop_pair == 3:
            # DDL-Ex3: Bỏ Pair 3 (Down3, Up2). Giữ Down1, Down2, Down4; Up1, Up3, Up4.
            # Encoder
            self.enc1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)       # 16 -> 32, H/2
            self.enc2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 32 -> 64, H/4
            self.enc3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 64 -> 128, H/8 (Bottleneck)
            # Decoder
            # dec1: input x3 (128) upsample H/8 -> H/4, concat skip x2 (64) => in_c = 128 + 64 = 192, out_c = 32
            self.dec1 = Up_repdw(base_c * 12, base_c * 2, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec2: input (32) upsample H/4 -> H/2, concat skip x1 (32) => in_c = 32 + 32 = 64, out_c = 16
            self.dec2 = Up_repdw(base_c * 4, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec3: input (16) upsample H/2 -> H, concat skip x0 (16) => in_c = 16 + 16 = 32, out_c = 16
            self.dec3 = Up_repdw(base_c * 2, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        elif drop_pair == 2:
            # DDL-Ex2: Bỏ Pair 2 (Down2, Up3). Giữ Down1, Down3, Down4; Up1, Up2, Up4.
            # Encoder
            self.enc1 = Down_repdw(base_c, base_c * 2, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)       # 16 -> 32, H/2
            self.enc2 = Down_repdw(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 32 -> 64, H/4
            self.enc3 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 64 -> 128, H/8 (Bottleneck)
            # Decoder
            # dec1: input x3 (128) upsample H/8 -> H/4, concat skip x2 (64) => in_c = 128 + 64 = 192, out_c = 32
            self.dec1 = Up_repdw(base_c * 12, base_c * 2, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec2: input (32) upsample H/4 -> H/2, concat skip x1 (32) => in_c = 32 + 32 = 64, out_c = 16
            self.dec2 = Up_repdw(base_c * 4, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec3: input (16) upsample H/2 -> H, concat skip x0 (16) => in_c = 16 + 16 = 32, out_c = 16
            self.dec3 = Up_repdw(base_c * 2, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        elif drop_pair == 1:
            # DDL-Ex1: Bỏ Pair 1 (Down1, Up4). Giữ Down2, Down3, Down4; Up1, Up2, Up3.
            # Encoder
            self.enc1 = Down_repdw(base_c, base_c * 4, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)       # 16 -> 64, H/2
            self.enc2 = Down_repdw(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 64 -> 128, H/4
            self.enc3 = Down_repdw(base_c * 8, base_c * 8, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)   # 128 -> 128, H/8 (Bottleneck)
            # Decoder
            # dec1: input x3 (128) upsample H/8 -> H/4, concat skip x2 (128) => in_c = 128 + 128 = 256, out_c = 64
            self.dec1 = Up_repdw(base_c * 16, base_c * 4, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec2: input (64) upsample H/4 -> H/2, concat skip x1 (64) => in_c = 64 + 64 = 128, out_c = 32
            self.dec2 = Up_repdw(base_c * 8, base_c * 2, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
            # dec3: input (32) upsample H/2 -> H, concat skip x0 (16) => in_c = 32 + 16 = 48, out_c = 16
            self.dec3 = Up_repdw(base_c * 3, base_c, bilinear=bilinear, deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)

        # Head: 1x1 Conv ra num_classes
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x):
        # Stem
        x0 = self.in_conv(x)      # (B, 16, H, W)

        # Encoder (3 stages)
        x1 = self.enc1(x0)        # (B, C1, H/2, W/2)
        x2 = self.enc2(x1)        # (B, C2, H/4, W/4)
        x3 = self.enc3(x2)        # (B, C3, H/8, W/8) - Bottleneck

        # Decoder (3 stages)
        d1 = self.dec1(x3, x2)    # Upsample to H/4, concat skip x2
        d2 = self.dec2(d1, x1)    # Upsample to H/2, concat skip x1
        d3 = self.dec3(d2, x0)    # Upsample to H, concat skip x0

        # Head
        logits = self.out_conv(d3) # (B, num_classes, H, W)
        return logits

    def switch_to_deploy(self):
        """
        Gập toàn bộ các khối multi-branch RepDW / RPD / RepConvbn về 1 nhánh duy nhất.
        """
        if getattr(self, 'deploy', False):
            return
        for m in self.modules():
            if hasattr(m, 'switch_to_deploy') and m is not self:
                m.switch_to_deploy()
        self.deploy = True
