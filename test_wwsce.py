"""Focused checks for the P2 RepDNet WWSCE data/loss path.

Run from the RPD_P2 root: ``python test_wwsce.py``.
"""

import torch
import torch.nn.functional as F

from datasets.pdc import build_size_weight_map
from models.losses import WeedSizeAwareCrossEntropy


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
    per_pixel_ce = F.cross_entropy(logits, target, reduction='none')
    expected_weights = torch.tensor([[[1.0, 3.0]]])
    expected = (per_pixel_ce * expected_weights).sum() / expected_weights.sum()
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


if __name__ == '__main__':
    test_size_weight_map()
    test_wwsce_formula_and_backward()
    print('WWSCE checks passed.')
