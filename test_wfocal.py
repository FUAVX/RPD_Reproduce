"""Focused checks for the P2 RepDWNet weighted focal-loss path.

Run from the RPD_P2 root: ``python test_wfocal.py``.
"""

import torch

from models.losses import WeightedFocalLoss


def baseline_weighted_ce(logits: torch.Tensor,
                         target: torch.Tensor,
                         class_weights: torch.Tensor,
                         mask_keep: torch.Tensor) -> torch.Tensor:
    """Manual, non-mutating form of the released RPD baseline WCE."""
    valid = mask_keep.bool()
    safe_target = target.clone()
    safe_target[~valid] = 0
    probabilities = torch.softmax(logits, dim=1)
    p_t = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    pixel_ce = -torch.log(torch.clamp(p_t, min=1e-12, max=1.0))
    weighted_loss = class_weights.to(logits)[safe_target] * pixel_ce
    return weighted_loss[valid].sum() / valid.sum()


def test_weighted_focal_formula_and_backward() -> None:
    logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]], [[-1.0, -1.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    criterion = WeightedFocalLoss([1.0, 2.0, 3.0], gamma=2.0)

    actual = criterion(logits, target, mode='train')
    probabilities = torch.softmax(logits, dim=1)
    p_t = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    pixel_ce = -torch.log(torch.clamp(p_t, min=1e-12, max=1.0))
    class_weight = torch.tensor([[[1.0, 2.0]]])
    expected = (class_weight * (1.0 - p_t).pow(2.0) * pixel_ce).sum() / target.numel()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_gamma_zero_recovers_baseline_wce() -> None:
    logits = torch.tensor([[[[2.0, -1.0, 0.2]],
                            [[0.0, 2.0, 0.1]],
                            [[-1.0, 0.0, 1.2]]]])
    target = torch.tensor([[[0, 255, 2]]])
    mask_keep = target != 255
    weights = torch.tensor([1.47, 5.06, 10.02])
    criterion = WeightedFocalLoss(weights.tolist(), gamma=0.0)

    actual = criterion(logits, target, mode='train', mask_keep=mask_keep)
    expected = baseline_weighted_ce(logits, target, weights, mask_keep)
    torch.testing.assert_close(actual, expected)


def test_ignored_pixels_do_not_change_loss_or_target() -> None:
    target = torch.tensor([[[0, 255, 2]]])
    target_before = target.clone()
    mask_keep = target != 255
    criterion = WeightedFocalLoss([1.47, 5.06, 10.02], gamma=2.0)

    logits_a = torch.tensor([[[[2.0, -30.0, 0.2]],
                              [[0.0, 30.0, 0.1]],
                              [[-1.0, 10.0, 1.2]]]])
    logits_b = logits_a.clone()
    logits_b[:, :, :, 1] = torch.tensor([100.0, -100.0, -100.0]).view(1, 3, 1)

    loss_a = criterion(logits_a, target, mode='train', mask_keep=mask_keep)
    loss_b = criterion(logits_b, target, mode='train', mask_keep=mask_keep)
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(target, target_before)


if __name__ == '__main__':
    test_weighted_focal_formula_and_backward()
    test_gamma_zero_recovers_baseline_wce()
    test_ignored_pixels_do_not_change_loss_or_target()
    print('WFocal checks passed.')
