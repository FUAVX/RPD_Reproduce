"""A-oA decoder unit ported from f/fuse-fpn; original block initialization retained."""
from torch import nn
from torch.nn import functional as F


class AddUp(nn.Module):
    """Keep Control's upsample/pad, Cmid and Cout; change fusion input width."""
    def __init__(self, channels, out_channels, projected, block, deploy, use_se, branches,
                 spatial_block='repdw'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Keep the existing initialization of the selected RepDW or B0-RO PDC block.
        self.conv = block(channels, out_channels, mid_channels=channels,
                          deploy=deploy, use_se=use_se, num_dw_branches=branches,
                          spatial_block=spatial_block)
        if projected:
            raise ValueError('DeDo supports A-oA without projection only')
        self.lateral = nn.Identity()

    def forward(self, decoder, skip):
        decoder = self.up(decoder)
        dy, dx = skip.shape[-2]-decoder.shape[-2], skip.shape[-1]-decoder.shape[-1]
        decoder = F.pad(decoder, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(decoder + self.lateral(skip))
