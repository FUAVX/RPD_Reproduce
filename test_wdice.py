"""Focused checks for the P2 RepDWNet WDice-loss path.

Run from the RPD_P2 root: ``python test_wdice.py``.
"""

import torch

from models.losses import WeightedCrossEntropyDiceLoss


def baseline_weighted_ce(logits: torch.Tensor, target: torch.Tensor,
                         class_weights: torch.Tensor, mask_keep: torch.Tensor) -> torch.Tensor:
    valid = mask_keep.bool()
    safe_target = target.clone()
    safe_target[~valid] = 0
    probabilities = torch.softmax(logits, dim=1)
    p_t = probabilities.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    pixel_ce = -torch.log(torch.clamp(p_t, min=1e-12, max=1.0))
    return (class_weights.to(logits)[safe_target][valid] * pixel_ce[valid]).sum() / valid.sum()


def test_wdice_formula_and_backward() -> None:
    logits = torch.tensor([[[[2.0, 0.0, -1.0]],
                            [[0.0, 2.0, -1.0]],
                            [[-1.0, -1.0, 2.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1, 2]]])
    weights = torch.tensor([1.0, 2.0, 3.0])
    criterion = WeightedCrossEntropyDiceLoss(weights.tolist(), dice_weight=0.5, epsilon=1e-6)

    actual = criterion(logits, target, mode='train')
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(target, num_classes=3).permute(0, 3, 1, 2).to(logits.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    expected = baseline_weighted_ce(logits, target, weights, torch.ones_like(target, dtype=torch.bool)) + 0.5 * dice_loss
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_zero_dice_weight_recovers_baseline_wce() -> None:
    logits = torch.tensor([[[[2.0, -1.0, 0.2]],
                            [[0.0, 2.0, 0.1]],
                            [[-1.0, 0.0, 1.2]]]])
    target = torch.tensor([[[0, 255, 2]]])
    mask_keep = target != 255
    weights = torch.tensor([1.47, 5.06, 10.02])
    criterion = WeightedCrossEntropyDiceLoss(weights.tolist(), dice_weight=0.0)

    actual = criterion(logits, target, mode='train', mask_keep=mask_keep)
    expected = baseline_weighted_ce(logits, target, weights, mask_keep)
    torch.testing.assert_close(actual, expected)


def test_ignored_pixels_do_not_change_loss_or_target() -> None:
    target = torch.tensor([[[0, 255, 2]]])
    target_before = target.clone()
    mask_keep = target != 255
    criterion = WeightedCrossEntropyDiceLoss([1.47, 5.06, 10.02])

    logits_a = torch.tensor([[[[2.0, -30.0, 0.2]],
                              [[0.0, 30.0, 0.1]],
                              [[-1.0, 10.0, 1.2]]]])
    logits_b = logits_a.clone()
    logits_b[:, :, :, 1] = torch.tensor([100.0, -100.0, -100.0]).view(1, 3, 1)

    loss_a = criterion(logits_a, target, mode='train', mask_keep=mask_keep)
    loss_b = criterion(logits_b, target, mode='train', mask_keep=mask_keep)
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(target, target_before)


def test_perfect_prediction_has_near_zero_dice_term() -> None:
    target = torch.tensor([[[0, 1, 2]]])
    logits = torch.full((1, 3, 1, 3), -20.0)
    logits.scatter_(1, target.unsqueeze(1), 20.0)
    weights = torch.tensor([1.0, 1.0, 1.0])
    criterion = WeightedCrossEntropyDiceLoss(weights.tolist(), dice_weight=1.0)

    total_loss = criterion(logits, target, mode='train')
    wce = baseline_weighted_ce(logits, target, weights, torch.ones_like(target, dtype=torch.bool))
    torch.testing.assert_close(total_loss - wce, torch.zeros_like(total_loss), atol=1e-5, rtol=0.0)


if __name__ == '__main__':
    test_wdice_formula_and_backward()
    test_zero_dice_weight_recovers_baseline_wce()
    test_ignored_pixels_do_not_change_loss_or_target()
    test_perfect_prediction_has_near_zero_dice_term()
    print('WDice checks passed.')
