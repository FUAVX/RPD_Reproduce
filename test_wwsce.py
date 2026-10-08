"""Focused checks for the P2 RepDNet WWSCE data/loss path.

Run from the RPD_P2 root: ``python test_wwsce.py``.
"""

import torch

from datasets.pdc import build_size_weight_map
from models.losses import WeedSizeAwareCrossEntropy


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
    return (pixel_ce[valid] * class_weights.to(logits)[safe_target][valid]).sum() / valid.sum()


def test_size_weight_map() -> None:
    anno = torch.tensor([
        [0, 2, 2, 0],
        [1, 2, 2, 2],
    ])
    instances = torch.tensor([
        [0, 7, 7, 0],
        [3, 9, 9, 9],
    ])
    result = build_size_weight_map(
        anno, instances, q25=2, q75=3,
        small_weight=1.5, medium_weight=1.25, large_weight=1.0)
    expected = torch.tensor([
        [1.0, 1.5, 1.5, 1.0],
        [1.0, 1.25, 1.25, 1.25],
    ])
    torch.testing.assert_close(result, expected)


def test_wwsce_formula_and_backward() -> None:
    logits = torch.tensor([[[[2.0, 0.0]], [[0.0, 2.0]], [[-1.0, -1.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1]]])
    size_weight_map = torch.tensor([[[1.0, 1.5]]])
    criterion = WeedSizeAwareCrossEntropy([1.0, 2.0, 3.0])

    actual = criterion(logits, target, mode='train', size_weight_map=size_weight_map)
    probabilities = torch.softmax(logits, dim=1)
    p_t = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
    per_pixel_ce = -torch.log(torch.clamp(p_t, min=1e-12, max=1.0))
    expected_weights = torch.tensor([[[1.0, 3.0]]])
    expected = (per_pixel_ce * expected_weights).sum() / target.numel()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_unit_size_weights_recover_baseline_wce() -> None:
    logits = torch.tensor([[[[2.0, -1.0, 0.2]],
                            [[0.0, 2.0, 0.1]],
                            [[-1.0, 0.0, 1.2]]]])
    target = torch.tensor([[[0, 255, 2]]])
    mask_keep = target != 255
    weights = torch.tensor([1.47, 5.06, 10.02])
    criterion = WeedSizeAwareCrossEntropy(weights.tolist())

    actual = criterion(
        logits, target, mode='train', mask_keep=mask_keep,
        size_weight_map=torch.ones_like(target, dtype=torch.float32))
    expected = baseline_weighted_ce(logits, target, weights, mask_keep)
    torch.testing.assert_close(actual, expected)


def test_ignored_pixels_do_not_change_loss_or_target() -> None:
    target = torch.tensor([[[0, 255, 2]]])
    target_before = target.clone()
    mask_keep = target != 255
    size_weight_map = torch.ones_like(target, dtype=torch.float32)
    criterion = WeedSizeAwareCrossEntropy([1.47, 5.06, 10.02])

    logits_a = torch.tensor([[[[2.0, -30.0, 0.2]],
                              [[0.0, 30.0, 0.1]],
                              [[-1.0, 10.0, 1.2]]]])
    logits_b = logits_a.clone()
    logits_b[:, :, :, 1] = torch.tensor([100.0, -100.0, -100.0]).view(1, 3, 1)

    loss_a = criterion(logits_a, target, mode='train', mask_keep=mask_keep,
                       size_weight_map=size_weight_map)
    loss_b = criterion(logits_b, target, mode='train', mask_keep=mask_keep,
                       size_weight_map=size_weight_map)
    torch.testing.assert_close(loss_a, loss_b)
    torch.testing.assert_close(target, target_before)


if __name__ == '__main__':
    test_size_weight_map()
    test_wwsce_formula_and_backward()
    test_unit_size_weights_recover_baseline_wce()
    test_ignored_pixels_do_not_change_loss_or_target()
    print('WWSCE checks passed.')
