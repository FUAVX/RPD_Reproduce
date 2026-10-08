import math
import pdb
from typing import List, Optional

import torch
from torch import nn


# ----------------------------------------------CROSS ENTROPY-------------------------------------
class CrossEntropy(nn.Module):
    def __init__(self, weights: Optional[List] = None):
        super(CrossEntropy, self).__init__()

        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
                mask_keep: Optional[torch.Tensor] = None,
                size_weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Compute cross entropy loss.

        Args:
            inputs(torch.Tensor): Unnormalized input tensor (logits) of shape [B x C x H x W]
            target(torch.Tensor): Ground-truth target tensor of shape [B x H x W]
            mode(str): train, val, or test
            mask_keep(Optional[torch.Tensor], optional): Mask of pixel of shape [B x H x W] which should be kept during
                                                loss computation (1: =keep, 0 :=ignore). Default to None := keep all.

        Returns:
            torch.Tensor: loss value (scalar)
        """
        assert mode in ['train', 'val', 'test']

        if mask_keep is not None:
            target[mask_keep == False] = 0

        # get the number classes and device
        batch_size, num_classes, height, width = inputs.shape
        input_device = inputs.device

        # convert logits to softmax probabilities
        probs = nn.functional.softmax(inputs, dim=1)  # [N x n_classes x H x W]
        del inputs

        # apply one-hot encoding to ground truth annotations
        target_one_hot = to_one_hot(target, int(num_classes))  # [N x n_classes x H x W]
        target_one_hot = target_one_hot.bool()
        del target

        # prepare to ignore certain pixels which should not be considered during loss computation
        if mask_keep is None:
            # consider all pixels to compute the loss
            mask_keep = torch.ones((batch_size, 1, height, width), dtype=torch.bool,
                                   device=input_device)  # [N x 1 x H x W]
        else:
            # get the dimension correctly
            mask_keep = mask_keep.unsqueeze(1)  # [N x 1 x H x W]

        # set ignore pixels to false
        target_one_hot = target_one_hot * mask_keep

        # gather the predicted probabilities of each ground truth category
        probs_gathered = probs[target_one_hot]  # M = N * (H * W) entries

        # make sure that probs are numerically stable when passed to log function: log(0) -> inf
        probs_gathered = torch.clip(probs_gathered, 1e-12, 1.0)

        # compute loss
        losses = -torch.log(probs_gathered)  # M = N * (H * W) entries
        del probs_gathered

        assert losses.shape[0] == torch.sum(mask_keep)
        del mask_keep

        # create weight matrix
        if self.weights is not None:
            if input_device != self.weights.device:
                self.weights = self.weights.to(input_device)

            weight_matrix = (target_one_hot.permute(0, 2, 3, 1) * self.weights).permute(0, 3, 1, 2)
            weight_gathered = weight_matrix[target_one_hot]
            assert torch.all(weight_gathered > 0)

            # compute weighted loss for each prediction
            losses *= weight_gathered

        return torch.mean(losses)


class WeedSizeAwareCrossEntropy(nn.Module):
    """Baseline-compatible weighted CE with a post-crop weed-size multiplier.

    The original RPD weighted CE averages weighted per-pixel losses over the
    number of valid pixels.  WWSCE preserves that normalization and changes
    only the numerator by applying the size multiplier to weed-instance pixels.
    """

    def __init__(self, weights: List[float], ignore_index: int = 255):
        super().__init__()
        if weights is None or len(weights) == 0:
            raise ValueError("WWSCE requires one positive class weight per class.")
        class_weights = torch.as_tensor(weights, dtype=torch.float32)
        if torch.any(class_weights <= 0):
            raise ValueError("WWSCE class weights must all be positive.")
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
                mask_keep: Optional[torch.Tensor] = None,
                size_weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f"Unknown loss mode: {mode}")
        if inputs.ndim != 4 or target.shape != inputs.shape[0:1] + inputs.shape[2:]:
            raise ValueError(f"Expected logits [B,C,H,W] and target [B,H,W], got {inputs.shape} and {target.shape}")
        if target.dtype != torch.long:
            target = target.long()

        # Keep the caller-defined mask semantics of the B1 control: training
        # supplies target != 255, while validation supplies no additional mask.
        # Do not mutate ``target`` while preparing the safe gather indices.
        if mask_keep is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            if mask_keep.shape != target.shape:
                raise ValueError(f"mask_keep shape {mask_keep.shape} does not match target {target.shape}")
            valid = mask_keep.bool()
        if not torch.any(valid):
            return inputs.sum() * 0.0

        if torch.any(valid & (target == self.ignore_index)):
            raise ValueError(
                f"Target contains ignore_index={self.ignore_index} on an unmasked pixel. "
                "Pass mask_keep for ignored labels, matching the B1 training path.")

        safe_target = target.clone()
        safe_target[~valid] = 0
        if torch.any(safe_target >= inputs.shape[1]) or torch.any(safe_target < 0):
            raise ValueError(
                "Target contains a class ID outside the logits range on an unmasked pixel. "
                "Pass mask_keep for ignored labels, matching the B1 training path.")

        # Match the released baseline implementation: softmax, gather the
        # ground-truth probability, clamp it, then take -log.
        probabilities = torch.softmax(inputs, dim=1)
        p_t = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        pixel_ce = -torch.log(torch.clamp(p_t, min=1e-12, max=1.0))
        class_weight = self.class_weights.to(device=inputs.device, dtype=inputs.dtype)[safe_target]
        if size_weight_map is None:
            size_weight_map = torch.ones_like(pixel_ce, dtype=inputs.dtype)
        else:
            if size_weight_map.shape != target.shape:
                raise ValueError(
                    f"size_weight_map shape {size_weight_map.shape} does not match target {target.shape}")
            size_weight_map = size_weight_map.to(device=inputs.device, dtype=inputs.dtype)
            if torch.any(size_weight_map[valid] <= 0):
                raise ValueError("WWSCE size weights must be positive for valid pixels.")

        effective_weight = class_weight * size_weight_map
        return (pixel_ce[valid] * effective_weight[valid]).sum() / valid.sum()


class WeightedFocalLoss(nn.Module):
    """Baseline-compatible class-weighted focal loss for the P2 Ex6 ablation.

    It preserves the original RPD weighted-CE normalization (divide by the
    number of caller-selected valid pixels) and adds only the focal factor.
    """

    def __init__(self, weights: List[float], gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        if weights is None or len(weights) == 0:
            raise ValueError("Weighted focal loss requires one positive class weight per class.")
        class_weights = torch.as_tensor(weights, dtype=torch.float32)
        if torch.any(class_weights <= 0):
            raise ValueError("Weighted focal loss class weights must all be positive.")
        if gamma < 0:
            raise ValueError(f"Expected gamma >= 0, got {gamma}.")
        self.register_buffer('class_weights', class_weights)
        self.gamma = float(gamma)
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
                mask_keep: Optional[torch.Tensor] = None,
                size_weight_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        del size_weight_map  # WFocal intentionally has no weed-size term.
        if mode not in ('train', 'val', 'test'):
            raise ValueError(f"Unknown loss mode: {mode}")
        if inputs.ndim != 4 or target.shape != inputs.shape[0:1] + inputs.shape[2:]:
            raise ValueError(f"Expected logits [B,C,H,W] and target [B,H,W], got {inputs.shape} and {target.shape}")
        if target.dtype != torch.long:
            target = target.long()

        # Match B1 caller semantics: train passes target != 255, while
        # validation applies no additional mask.
        if mask_keep is None:
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            if mask_keep.shape != target.shape:
                raise ValueError(f"mask_keep shape {mask_keep.shape} does not match target {target.shape}")
            valid = mask_keep.bool()
        if not torch.any(valid):
            return inputs.sum() * 0.0
        if torch.any(valid & (target == self.ignore_index)):
            raise ValueError(
                f"Target contains ignore_index={self.ignore_index} on an unmasked pixel. "
                "Pass mask_keep for ignored labels, matching the B1 training path.")

        safe_target = target.clone()
        safe_target[~valid] = 0
        if torch.any(safe_target >= inputs.shape[1]) or torch.any(safe_target < 0):
            raise ValueError(
                "Target contains a class ID outside the logits range on an unmasked pixel. "
                "Pass mask_keep for ignored labels, matching the B1 training path.")

        probabilities = torch.softmax(inputs, dim=1)
        p_t = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
        p_t = torch.clamp(p_t, min=1e-12, max=1.0)
        pixel_ce = -torch.log(p_t)
        focal_factor = torch.pow(1.0 - p_t, self.gamma)
        class_weight = self.class_weights.to(device=inputs.device, dtype=inputs.dtype)[safe_target]
        weighted_loss = class_weight * focal_factor * pixel_ce
        return weighted_loss[valid].sum() / valid.sum()


# -------------------------------------Generalized-Jensen-Shannon Divergence -------------------------------------------
def gjs_div_loss(p1_logits: torch.Tensor, p2_logits: torch.Tensor, p3_logits: torch.Tensor) -> torch.Tensor:
    p1_probs = nn.functional.softmax(p1_logits, dim=1)  # [BxCxHxW]
    p2_probs = nn.functional.softmax(p2_logits, dim=1)
    p3_probs = nn.functional.softmax(p3_logits, dim=1)

    m_probs = (p1_probs + p2_probs + p3_probs) / 3.0  # [B x C x H x W]
    m_probs = torch.clamp(m_probs, 1e-7, 1.0).log()

    loss1 = nn.functional.kl_div(input=m_probs, target=p1_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss1 = torch.sum(loss1, dim=1)  # [B x H x W]

    loss2 = nn.functional.kl_div(input=m_probs, target=p2_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss2 = torch.sum(loss2, dim=1)  # [B x H x W]

    loss3 = nn.functional.kl_div(input=m_probs, target=p3_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss3 = torch.sum(loss3, dim=1)  # [B x H x W]

    loss = (loss1 + loss2 + loss3) / 3.0  # [B x H x W]
    loss = loss.mean()

    return loss


# -----------------------------------------Jensen-Shannon Divergence ---------------------------------------------------
def js_div_loss(p_logits: torch.Tensor, q_logits: torch.Tensor,
                mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute Jensen-Shannon divergence.

    Args:
        p(torch.Tensor): 1st distributions of shape [B x C x H x W]
        q(torch.Tensor): 2nd distributions of shape [B x C x H x W]
        mask_keep(Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss
                                                    during loss computation (1 := keep, 0 :=ignore). Defaults to None.

    Returns:
          torch.Tensor: loss value
    """
    p_probs = nn.functional.softmax(p_logits, dim=1)
    q_probs = nn.functional.softmax(q_logits, dim=1)
    m_probs = (p_probs + q_probs) * 0.5

    p_probs = torch.clamp(p_probs, 1e-12, 1)
    q_probs = torch.clamp(q_probs, 1e-12, 1)
    m_probs = torch.clamp(m_probs, 1e-12, 1)

    kl_p_m = p_probs * torch.log(p_probs / m_probs)  # [B, C, H, W]
    kl_p_m = torch.sum(kl_p_m, dim=1)  # [B, H, W]

    kl_q_m = q_probs * torch.log(q_probs / m_probs)  # [B, C, H, W]
    kl_q_m = torch.sum(kl_q_m, dim=1)  # [B, H, W]

    # compute Jensen_Shannon divergence
    js_p_q = (0.5 * kl_p_m) + (0.5 * kl_q_m)  # [B, H, W]

    if mask_keep is not None:
        # [M] where M is number of pixel which should be kept according to mask_keep (i.e., torch.sum(mask_keep)=M)
        js_p_q = js_p_q[mask_keep]

    loss = torch.mean(js_p_q)

    assert loss >= 0, f"Invalid loss for js divergence: {loss}"  # lower bound
    assert loss <= math.log(2), f"Invalid loss for js divergence: {loss}"  # upper bound

    return loss


# ------------------------------------------ Kullback–Leibler Divergence -----------------------------------------------
def kl_div_loss(x_logits_pred: torch.Tensor, x_logits_true: torch.Tensor,
                mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute KL-Divergence.

    There are difference ways to compute the Kullback-Leibler Divergence.
    We refer to https://machinelearningmastery.com/divergence-between-probability-distributions/ for more information.

    Args:
        x_logits_pred(torch.Tensor): Source distributions of shape [B x C x H x W]
        x_logits_true(torch.Tensor): Target distributions of shape [B x C x H x W]
        mask_keep(Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during
                                                     loss computation (1 :=keep, 0 :=ignore). Defaults to None
    """
    x_pred = nn.functional.softmax(x_logits_pred, dim=1)
    x_true = nn.functional.softmax(x_logits_true, dim=1)

    x_pred = torch.clamp(x_pred, 1e-12, 1)
    x_true = torch.clamp(x_true, 1e-12, 1)

    loss = x_true * torch.log(x_true) / (x_pred)  # [B x C x H x W]
    loss = torch.sum(loss, dim=1)  # [B x H x W]

    if mask_keep is not None:
        loss = loss[mask_keep]

    loss = torch.mean(loss)

    assert loss >= 0, f"Invalid loss for kl divergence: {loss}"

    return loss


# ------------------------------------------UTILS-----------------------------------------------------------------------
def get_div_loss_weight():
    pass


def to_one_hot(tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    """ Convert tensor to its one hot encoded version.

    Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

    Args:
        tensor(torch.Tensor): ground truth tensor of shape [N x n_classes x H x W]
    """
    if len(tensor.size()) == 1:
        b = tensor.size(0)
        if tensor.is_cuda:
            one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 2:
        n, b = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 3:
        n, h, w = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot


def get_criterion(cfg) -> nn.Module:
    loss_name = cfg['train']['loss']

    if loss_name == 'xentropy':
        weights = cfg['train']['class_weights']
        return CrossEntropy(weights)

    if loss_name == 'weed_size_aware_ce':
        loss_cfg = cfg.get('loss', {})
        ignore_index = int(loss_cfg.get('ignore_index', 255))
        return WeedSizeAwareCrossEntropy(cfg['train']['class_weights'], ignore_index=ignore_index)

    if loss_name == 'weighted_focal':
        loss_cfg = cfg.get('loss', {})
        focal_cfg = loss_cfg.get('focal', {})
        gamma = float(focal_cfg.get('gamma', 2.0))
        ignore_index = int(loss_cfg.get('ignore_index', 255))
        return WeightedFocalLoss(cfg['train']['class_weights'], gamma=gamma, ignore_index=ignore_index)

    raise ValueError(f"Unsupported loss: {loss_name}")

