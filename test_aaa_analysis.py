"""WSC physical coordinates, preprocessing, coverage and RNG isolation tests."""
import random
import unittest
import numpy as np
import torch
from torch import nn
from scripts.analyze_aaa import (OFFSETS, aligned_slices, wsc_pair, aggregate_wsc,
                                  remap_semantics, evaluate_normalized_image)
from scripts.benchmark_aaa import isolated_rng


class WSCAnalysisTests(unittest.TestCase):
    def test_all_direction_physical_alignment(self):
        physical = torch.arange(8*8).reshape(8,8).float()/64
        base = physical[1:7,1:7]
        for name, (dy,dx) in OFFSETS.items():
            view = physical[1+dy:7+dy,1+dx:7+dx]
            gt = torch.full((6,6),2)
            row = wsc_pair(base, view, gt, (dy,dx))
            self.assertEqual(row['wsc'],1,name)
            self.assertEqual(row['weed_pixels'],30)
            a,b=aligned_slices((dy,dx),(6,6));self.assertTrue(torch.equal(base[a],view[b]))

    def test_gt_weed_roi_ignore_and_empty_cases(self):
        base = torch.zeros(4,4);view = torch.ones(4,4)
        gt = torch.full((4,4),255);gt[2,2]=2
        r=wsc_pair(base,view,gt,(0,1))
        self.assertEqual(r['weed_pixels'],1);self.assertEqual(r['wsc'],0)
        gt[2,2]=1
        self.assertIsNone(wsc_pair(base,view,gt,(0,1))['wsc'])
        torch.testing.assert_close(remap_semantics(np.array([[3,4,255]])),torch.tensor([[1,2,255]]))
        with self.assertRaises(ValueError):remap_semantics(np.array([[7]]))
        with self.assertRaises(ValueError):wsc_pair(base,view*2,gt,(0,1))

    def test_macro_images_not_pixels_or_pairs_and_empty_coverage(self):
        rows=[{'image':'a','direction':d,'wsc':1.,'weed_pixels':10000} for d in OFFSETS]
        rows += [{'image':'b','direction':d,'wsc':0. if d=='up' else None,'weed_pixels':1 if d=='up' else 0} for d in OFFSETS]
        rows += [{'image':'empty','direction':d,'wsc':None,'weed_pixels':0} for d in OFFSETS]
        _,summary=aggregate_wsc(rows)
        self.assertEqual(summary['mean_wsc'],.5)
        self.assertEqual(summary['std_wsc_across_images'],.5)
        self.assertEqual(summary['evaluated_images'],2)
        self.assertEqual(summary['skipped_images'],1)
        self.assertEqual(summary['skipped_image_offset_pairs'],7)
        self.assertEqual(summary['per_direction']['up']['mean_wsc'],.5)
        _,empty=aggregate_wsc(rows[-4:]);self.assertIsNone(empty['mean_wsc'])

    def test_normalize_full_once_pointwise_model_is_consistent(self):
        from datasets.augmentations_normalize import SingleImageNormalizer
        g=torch.Generator().manual_seed(1682409321)
        x=torch.rand(3,1024,1024,generator=g);x[:,:,0:16]+=2
        normalizer=SingleImageNormalizer();full=normalizer.normalize(x)
        torch.testing.assert_close(full[:,8:1016,9:1016],full[:,8:1016,9:1017][:,:,:-1],atol=0,rtol=0)
        a=normalizer.normalize(x[:,8:1016,8:1016]);b=normalizer.normalize(x[:,8:1016,9:1017])
        self.assertGreater((a[:,:,1:]-b[:,:,:-1]).abs().max().item(),0)
        class Pointwise(nn.Module):
            def forward(self,v):return v
        model=Pointwise().eval()
        rows=evaluate_normalized_image(model,full,torch.full((1024,1024),2),torch.device('cpu'))
        self.assertEqual(len(rows),4)
        for r in rows:self.assertEqual(r['wsc'],1)
        model.train()
        with self.assertRaises(ValueError):evaluate_normalized_image(model,full,torch.full((1024,1024),2),'cpu')

    def test_disposable_benchmark_restores_random_streams_even_on_error(self):
        torch.manual_seed(1682409321);np.random.seed(1);random.seed(1)
        cpu=torch.get_rng_state().clone();np_state=np.random.get_state();py=random.getstate()
        with self.assertRaises(RuntimeError):
            with isolated_rng(1682409321,'cpu'):
                torch.randn(50);np.random.rand(20);random.random();raise RuntimeError('probe')
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        now=np.random.get_state()
        self.assertEqual(np_state[0],now[0]);np.testing.assert_array_equal(np_state[1],now[1]);self.assertEqual(np_state[2:],now[2:])
        self.assertEqual(py,random.getstate())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
