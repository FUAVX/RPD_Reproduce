import torch
from torch import nn
import copy
from models.rpdnet.RPD_Module import *


def conv_bn_act(in_channels, out_channels, kernel_size, stride, padding, groups):
    if padding is None:
        padding = kernel_size // 2
    mod_list = nn.Sequential()
    mod_list.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups, bias=False))
    mod_list.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    mod_list.add_module('act', nn.ReLU())
    return mod_list


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class RepConvbn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, deploy=False):
        super(RepConvbn, self).__init__()

        self.deploy = deploy

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = nn.ReLU(inplace=True)

        padding = kernel_size // 2

        if deploy:
            self.cba_rep = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias=True)
        else:
            self.cba_ori = conv_bn_act(in_channels, out_channels, kernel_size, stride, padding, groups)

    def forward(self, inputs):
        if hasattr(self, 'cba_rep'):
            out = self.activation(self.cba_rep(inputs))
        else:
            out = self.cba_ori(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.cba_ori.conv, self.cba_ori.bn)
        return eq_k, eq_b

    def switch_to_deploy(self):
        if hasattr(self, 'cba_rep'):
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.cba_rep = nn.Conv2d(in_channels=self.cba_ori.conv.in_channels,
                                 out_channels=self.cba_ori.conv.out_channels,
                                 kernel_size=self.cba_ori.conv.kernel_size,
                                 stride=self.cba_ori.conv.stride,
                                 padding=self.cba_ori.conv.padding,
                                 groups=self.cba_ori.conv.groups)
        self.cba_rep.weight.data = eq_k
        self.cba_rep.bias.data = eq_b
        self.__delattr__('cba_ori')


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, deploy=False, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            RepConvbn(in_channels, mid_channels, kernel_size, stride=stride, groups=groups, deploy=deploy),
            RepConvbn(mid_channels, out_channels, kernel_size, stride=stride, groups=groups, deploy=deploy)
        )


class DoubleConv_down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, deploy=False, use_se=False, convert=False):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv_down, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.Conv1 = self._make_stage(in_channels, mid_channels, deploy, use_se, convert)
        self.Conv2 = self._make_stage(mid_channels, out_channels, deploy, use_se, convert)

    def forward(self, input):
        x = self.Conv1(input)
        x = self.Conv2(x)

        return x

    def _make_stage(self, in_channels, out_channels, deploy, use_se, convert):
        rpd_layers = []
        rpd_layers.append(RPD('pdc', in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1,
                              groups=in_channels, bias=True, deploy=deploy, use_se=use_se, convert=convert,
                              num_conv_branches=1))
        rpd_layers.append(RPD('cv', in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                              groups=1, bias=True, deploy=deploy, use_se=use_se, convert=convert, num_conv_branches=4))

        return nn.Sequential(*rpd_layers)


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, deploy=False, use_se=False, convert=False):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv_down(in_channels, out_channels, deploy=deploy, use_se=use_se, convert=convert)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, deploy=False, use_se=False, convert=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv_down(in_channels, out_channels, in_channels // 2, deploy=deploy, use_se=use_se,
                                        convert=convert)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv_down(in_channels, out_channels, deploy=deploy, use_se=use_se, convert=convert)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                    diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)

        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class RPDNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 16,
                 deploy=False,
                 use_se=False,
                 convert=False):
        super(RPDNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c, deploy=deploy)
        self.down1 = Down(base_c, base_c * 2, deploy=deploy, use_se=use_se, convert=convert)
        self.down2 = Down(base_c * 2, base_c * 4, deploy=deploy, use_se=use_se, convert=convert)
        self.down3 = Down(base_c * 4, base_c * 8, deploy=deploy, use_se=use_se, convert=convert)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor, deploy=deploy, use_se=use_se, convert=convert)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear, deploy=deploy, use_se=use_se, convert=convert)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear, deploy=deploy, use_se=use_se, convert=convert)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear, deploy=deploy, use_se=use_se, convert=convert)
        self.up4 = Up(base_c * 2, base_c, bilinear, deploy=deploy, use_se=use_se, convert=convert)
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

