"""Fresh En/De ablations using the exact B1 Run 5 block implementations."""
import torch.nn as nn

from .RepDWNet import RepDWNet, DoubleConv, Down_repdw, Up_repdw, DoubleConv_down_repdw, OutConv


class NotebookUp(Up_repdw):
    """Keep B1 upsampling/padding/concat forward with an explicit mid width."""
    def __init__(self, in_channels, out_channels, mid_channels, **settings):
        nn.Module.__init__(self)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv_down_repdw(
            in_channels, out_channels, mid_channels=mid_channels, **settings)


class RepDWNetEnDe(RepDWNet):
    def __init__(self, *, num_classes, pretrained=False, deploy=False, use_se=False,
                 base_c=16, num_dw_branches=4, num_pw_branches=4, bilinear=True,
                 stem_output_channels=16, encoder_output_channels,
                 decoder_concat_channels, decoder_mid_channels,
                 decoder_output_channels, head_in_channels):
        nn.Module.__init__(self)
        if (pretrained or deploy or use_se or base_c != 16 or not bilinear
                or stem_output_channels != 16 or num_dw_branches != 4
                or num_pw_branches != 4 or num_classes != 3):
            raise ValueError('En/De must retain B1 Run 5 blocks and fresh training settings.')
        actual = (list(encoder_output_channels), list(decoder_concat_channels),
                  list(decoder_mid_channels), list(decoder_output_channels), head_in_channels)
        if actual != self.expected_channels:
            raise ValueError(f'{type(self).__name__}: channel table differs from handoff: {actual}')
        self.in_channels, self.num_classes = 3, num_classes
        self.bilinear, self.base_c = bilinear, base_c
        self.num_dw_branches, self.num_pw_branches = num_dw_branches, num_pw_branches
        self.channel_table = dict(stem=16, encoder=actual[0], decoder_concat=actual[1],
                                  decoder_mid=actual[2], decoder_output=actual[3],
                                  head_input=head_in_channels, head_output=num_classes)
        settings = dict(deploy=deploy, use_se=use_se, num_dw_branches=num_dw_branches)
        self.in_conv = DoubleConv(3, 16, deploy=deploy)
        previous = 16
        for i, output in enumerate(encoder_output_channels, 1):
            setattr(self, f'down{i}', Down_repdw(previous, output, **settings))
            previous = output
        for i, (concat, mid, output) in enumerate(zip(
                decoder_concat_channels, decoder_mid_channels, decoder_output_channels), 1):
            setattr(self, f'up{i}', NotebookUp(concat, output, mid_channels=mid, **settings))
        self.out_conv = OutConv(head_in_channels, num_classes)
        # Forward is inherited unchanged from B1 RepDWNet; identity stays conditional.


class NotebookRepDWNetExp1(RepDWNetEnDe):
    expected_channels = ([16, 32, 64, 64], [128, 96, 48, 32],
                         [128, 64, 32, 16], [64, 32, 16, 16], 16)


class NotebookRepDWNetExp2(RepDWNetEnDe):
    expected_channels = ([32, 64, 128, 128], [256, 96, 48, 24],
                         [64, 32, 16, 8], [32, 16, 8, 8], 8)
