"""Synthetic ownership, ignore handling and offline hook regression checks."""
import math
import unittest
import torch
from torch import nn
from scripts.analyze_lip import analyze_stage, summarize_windows, aggregate_images, LIPCapture, window_view
from models.rpdnet.lip import BottleneckLIP


class AnalysisTests(unittest.TestCase):
    def test_alignment_single_pixel_all_stages(self):
        mask = torch.zeros(32, 32, dtype=torch.long)
        mask[0, 0] = 2
        for stage in (1, 2, 3, 4):
            t = 2 ** (stage - 1)
            z = torch.zeros(1, 2, 32//t, 32//t)
            m = analyze_stage(z, mask, stage)
            self.assertEqual(m['mixed_windows'], 1)
            torch.testing.assert_close(m['occupancy'], torch.tensor([1/(4*t*t)]))
            torch.testing.assert_close(m['enrichment'], torch.ones(1))
            torch.testing.assert_close(m['entropy'], torch.ones(1))

    def test_nonuniform_E_one_and_entropy_per_channel(self):
        mask = torch.tensor([[2, 0], [0, 0]])
        w = torch.tensor([[[[.25, .65], [.05, .05]], [[.25, .05], [.65, .05]]]])
        m = analyze_stage(w.log(), mask, 1)
        torch.testing.assert_close(m['enrichment'], torch.ones(1))
        entropy = -torch.xlogy(w, w).sum((2, 3)).mean()/math.log(4)
        torch.testing.assert_close(m['entropy'][0], entropy)
        self.assertLess(float(m['entropy'][0]), 1)
        mean_w = w.mean(1)
        self.assertFalse(torch.isclose(entropy, -torch.xlogy(mean_w, mean_w).sum()/math.log(4)))

    def test_equal_occupancy_and_ignore(self):
        mask = torch.tensor([[2, 0, 2, 0], [0, 0, 0, 0],
                             [2, 0, 2, 0], [0, 0, 0, 0]])
        z = torch.tensor([[[[0., 1.], [2., 3.]]]])
        m = analyze_stage(z, mask, 2)
        self.assertTrue(m['equal_occupancy'].item())
        torch.testing.assert_close(m['enrichment'], torch.ones(1))
        mask[3, 3] = 255
        m = analyze_stage(z, mask, 2)
        self.assertEqual(m['invalid_windows'], 1)
        self.assertEqual(m['mixed_windows'], 0)
        self.assertIsNone(summarize_windows(m)['all_mixed']['enrichment'])

    def test_odd_border_and_spatial_alignment(self):
        mask = torch.zeros(5, 7, dtype=torch.long)
        mask[-1, -1] = 2
        m = analyze_stage(torch.zeros(1, 2, 5, 7), mask, 1)
        self.assertEqual(m['total_windows'], 6)
        self.assertEqual(m['mixed_windows'], 0)
        with self.assertRaises(ValueError):
            analyze_stage(torch.zeros(1, 2, 4, 4), mask, 1)

    def test_equal_image_weighting_and_empty_images(self):
        rows = []
        for name, mask, z in [('one', torch.tensor([[2,0],[0,0]]), torch.zeros(1,1,2,2)),
                              ('two', torch.tensor([[2,0],[0,0]]), torch.tensor([[[[3.,0.],[0.,0.]]]])),
                              ('empty', torch.zeros(2,2,dtype=torch.long), torch.zeros(1,1,2,2))]:
            rows.append({'image': name, 'stage': 1, **summarize_windows(analyze_stage(z,mask,1))})
        summary = aggregate_images(rows)['1']
        self.assertEqual(summary['images_without_mixed_windows'], 1)
        expected = sum(row['all_mixed']['enrichment']['mean'] for row in rows[:2])/2
        self.assertAlmostEqual(summary['all_mixed']['enrichment_image_mean'], expected)

    def test_hooks_leave_forward_and_cleanup_unchanged(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                for stage in range(1,5):
                    setattr(self, f'down{stage}', nn.Sequential(BottleneckLIP(4) if stage==1 else nn.Identity()))
            def forward(self,x):
                return self.down1(x)
        model = Toy().eval()
        x = torch.randn(1,4,6,8)
        before = model(x)
        with LIPCapture(model) as capture:
            actual = model(x)
            torch.testing.assert_close(actual, before, rtol=0, atol=0)
            self.assertFalse(capture.logits[1].requires_grad)
            weights = window_view(capture.logits[1]).softmax(-1)
            torch.testing.assert_close((window_view(x)*weights).sum(-1), actual)
        self.assertFalse(model.down1[0].logit.gate._forward_hooks)
        try:
            with LIPCapture(model):
                raise RuntimeError('intentional')
        except RuntimeError:
            pass
        self.assertFalse(model.down1[0].logit.gate._forward_hooks)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
