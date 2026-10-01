import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_c, out_c, kernel_size, stride=1, padding=0, groups=1):
    """Tạo Sequential(nn.Conv2d + nn.BatchNorm2d) chuẩn PyTorch."""
    mod = nn.Sequential()
    mod.add_module('conv', nn.Conv2d(in_c, out_c, kernel_size,
                                     stride=stride, padding=padding,
                                     groups=groups, bias=False))
    mod.add_module('bn', nn.BatchNorm2d(out_c))
    return mod


class SEblock(nn.Module):
    """Squeeze-and-Excitation độc lập cho RepDW."""
    def __init__(self, in_channels, squeeze_factor=16):
        super().__init__()
        squeeze_channels = max(8, in_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = F.hardsigmoid(self.fc2(scale), inplace=True)
        return scale * x


class RepDW(nn.Module):
    """
    Re-parameterizable Depthwise Convolution Block (RepDW).
    
    Thay thế toàn bộ toán tử vi sai PDC (cd, ad, rd) bằng các nhánh
    Depthwise Conv 3x3 chuẩn kết hợp 1x1 và Identity.
    Khi deploy, toàn bộ các nhánh tuyến tính được chập về 1 Conv2d duy nhất.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 groups=None, bias=True, deploy=False, use_se=False,
                 num_dw_branches=4):
        super().__init__()
        if groups is None:
            groups = in_channels

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        self.groups = groups
        self.bias = bias
        self.use_se = use_se
        self.num_dw_branches = num_dw_branches

        self.activation = nn.ReLU()
        self.se = SEblock(out_channels, squeeze_factor=16) if use_se else nn.Identity()

        if deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=self.padding,
                                      groups=groups, bias=bias)
        else:
            # Nhánh Identity: chỉ tồn tại khi kích thước kênh và spatial không đổi
            self.rbr_identity = (nn.BatchNorm2d(in_channels)
                                 if out_channels == in_channels and stride == 1
                                 else None)

            # Nhánh 1x1: chỉ tồn tại khi kernel_size > 1
            self.rbr_1x1 = (_conv_bn(in_channels, out_channels, 1,
                                     stride=stride, padding=0, groups=groups)
                            if kernel_size > 1 else None)

            # M nhánh Depthwise Conv 3x3 song song (đặt tên rbr_dw để cô lập với RPD)
            self.rbr_dw = nn.ModuleList([
                _conv_bn(in_channels, out_channels, kernel_size,
                         stride=stride, padding=self.padding, groups=groups)
                for _ in range(num_dw_branches)
            ])

    def forward(self, inputs):
        if hasattr(self, 'rep_conv'):
            return self.se(self.activation(self.rep_conv(inputs)))

        identity_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0
        point_out = self.rbr_1x1(inputs) if self.rbr_1x1 is not None else 0

        out = identity_out + point_out
        for ix in range(len(self.rbr_dw)):
            out = out + self.rbr_dw[ix](inputs)

        return self.se(self.activation(out))

    def _fuse_bn_tensor(self, branch):
        """
        Fuse Conv+BN hoặc Identity BN về cặp (kernel, bias) tương đương.
        Tính toán ở float32 để chống tràn số dưới Mixed Precision (AMP).
        Tạo Dirac delta tensor cục bộ theo device/dtype của bn.weight.
        """
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            bn = branch.bn
        else:  # nn.BatchNorm2d (nhánh Identity)
            bn = branch
            input_dim = self.in_channels // self.groups
            kernel = torch.zeros(
                self.in_channels, input_dim, self.kernel_size, self.kernel_size,
                dtype=bn.weight.dtype, device=bn.weight.device
            )
            for i in range(self.in_channels):
                kernel[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1.0

        target_dtype = kernel.dtype
        target_device = kernel.device

        k = kernel.float()
        running_mean = bn.running_mean.float()
        running_var = bn.running_var.float()
        gamma = bn.weight.float()
        beta = bn.bias.float()
        eps = float(bn.eps)

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        fused_k = (k * t).to(dtype=target_dtype, device=target_device)
        fused_b = (beta - running_mean * gamma / std).to(dtype=target_dtype, device=target_device)

        return fused_k, fused_b

    def _get_equivalent_kernel_bias(self):
        kernel_1x1, bias_1x1 = 0, 0
        if self.rbr_1x1 is not None:
            k1, b1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernel_1x1 = F.pad(k1, [self.kernel_size // 2] * 4)
            bias_1x1 = b1

        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)

        kernel_dw, bias_dw = 0, 0
        for ix in range(len(self.rbr_dw)):
            k, b = self._fuse_bn_tensor(self.rbr_dw[ix])
            kernel_dw = kernel_dw + k
            bias_dw = bias_dw + b

        return (kernel_dw + kernel_1x1 + kernel_id,
                bias_dw + bias_1x1 + bias_id)

    @torch.no_grad()
    def switch_to_deploy(self):
        """Chuyển đổi sang mô hình suy luận 1-pass, giải phóng các nhánh phụ."""
        if hasattr(self, 'rep_conv'):
            return

        eq_k, eq_b = self._get_equivalent_kernel_bias()

        self.rep_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=True
        ).to(device=eq_k.device, dtype=eq_k.dtype)

        self.rep_conv.weight.copy_(eq_k)
        self.rep_conv.bias.copy_(eq_b)

        if hasattr(self, 'rbr_dw'):
            self.__delattr__('rbr_dw')
        if hasattr(self, 'rbr_1x1'):
            self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

        self.deploy = True
