"""Lite-DPPSym port of the authors' nngraph implementation (not the CUDA variant).

Reference: visinf/dpp, commit e810bd63ed61bd7710559467bcd9a35ab9a1b028,
visinf_package/{DPP_sym_lite,CWeightCalc,PositiveBias}.lua.
"""

import math

import torch
from torch import nn


class LiteDPPSym(nn.Module):
    def __init__(self, channels, epsilon_squared=0.001, alpha_init=1.0,
                 lambda_init=1.0):
        super().__init__()
        if channels < 1 or min(epsilon_squared, alpha_init, lambda_init) <= 0:
            raise ValueError('DPP channels and epsilon_squared/alpha_init/lambda_init must be positive.')
        self.channels = channels
        self.epsilon_squared = float(epsilon_squared)
        # Match the authors' PositiveBias.lua / CWeightCalc.lua reset():
        # https://github.com/visinf/dpp/tree/e810bd63ed61bd7710559467bcd9a35ab9a1b028/visinf_package
        # With default alpha_init=lambda_init=1, both log vectors are U(-1/sqrt(C), 1/sqrt(C)).
        # This follows the released code; the paper describes Gaussian perturbations instead.
        bound = 1.0 / math.sqrt(channels)
        self.log_alpha = nn.Parameter(torch.empty(channels).uniform_(-bound, bound).add_(math.log(alpha_init)))
        self.log_lambda = nn.Parameter(torch.empty(channels).uniform_(-bound, bound).add_(math.log(lambda_init)))

    @property
    def alpha(self):
        return self.log_alpha.clamp(max=10).exp()

    @property
    def exponent(self):
        return self.log_lambda.exp()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(f'Expected [B,{self.channels},H,W], got {tuple(x.shape)}')
        b, c, height, width = x.shape
        h, w = height // 2, width // 2
        if h == 0 or w == 0:
            raise ValueError('DPP needs spatial dimensions >= 2.')
        # Match MaxPool2d(2,2): discard the last row/column for odd dimensions.
        # [B,C,H/2,W/2,4], keeping all four values of each local window.
        windows = x[:, :, :2*h, :2*w].reshape(b, c, h, 2, w, 2)
        windows = windows.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w, 4)
        difference = windows - windows.mean(dim=-1, keepdim=True)
        energy = difference.square() + self.epsilon_squared
        # DPP_sym_lite.lua divides energy by its 2x2 local average before CWeightCalc.
        # Subtracting logs is algebraically the same normalization, including gradients.
        log_reward = energy.log() - energy.mean(dim=-1, keepdim=True).log()
        log_reward = log_reward * (0.5 * self.exponent.view(1, c, 1, 1, 1))
        # PositiveBias.lua projects log-alpha to <=10 before exp, while retaining
        # the gradient at that projected value. Only mutate when projection is needed
        # so repeated forwards below the cap do not invalidate autograd graphs.
        with torch.no_grad():
            if torch.any(self.log_alpha > 10):
                self.log_alpha.clamp_(max=10)
        # logaddexp + softmax computes normalized (alpha + reward) without
        # forming large reward powers. No detached statistics enter forward.
        log_weights = torch.logaddexp(self.log_alpha.view(1, c, 1, 1, 1), log_reward)
        weights = log_weights.softmax(dim=-1)
        return (weights * windows).sum(dim=-1)


def make_downsampling(channels, config=None, stage=None):
    if config is None:
        return nn.MaxPool2d(2, stride=2)
    allowed = {'variant', 'stages', 'epsilon_squared', 'alpha_init', 'lambda_init'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'Unknown downsampling settings: {sorted(unknown)}')
    variant = config.get('variant', 'maxpool')
    if variant not in ('maxpool', 'lite_dpp_sym'):
        raise ValueError(f'Unsupported downsampling variant: {variant!r}. Currently supported: maxpool, lite_dpp_sym.')
    stages = config.get('stages', [])
    if not isinstance(stages, (list, tuple)) or any(s not in (1, 2, 3, 4) for s in stages):
        raise ValueError('downsampling.stages must contain stage numbers 1..4.')
    if len(set(stages)) != len(stages):
        raise ValueError('Duplicate downsampling stages.')
    if variant == 'maxpool' and stages:
        raise ValueError('maxpool requires an empty stages list.')
    if variant == 'lite_dpp_sym' and not stages:
        raise ValueError('lite_dpp_sym requires at least one stage.')
    if stage not in stages:
        return nn.MaxPool2d(2, stride=2)
    kwargs = {key: config[key] for key in allowed - {'variant', 'stages'} if key in config}
    return LiteDPPSym(channels, **kwargs)


def dpp_optimizer_groups(model, learning_rate, weight_decay):
    """Keep the P2 Adam group unchanged; exclude new DPP log parameters from decay."""
    dpp_ids = {id(p) for module in model.modules() if isinstance(module, LiteDPPSym)
               for p in module.parameters()}
    regular, dpp = [], []
    for parameter in model.parameters():
        (dpp if id(parameter) in dpp_ids else regular).append(parameter)
    groups = [{'params': regular, 'lr': learning_rate, 'weight_decay': weight_decay}]
    if dpp:
        groups.append({'params': dpp, 'lr': learning_rate, 'weight_decay': 0.0})
    return groups
