"""Meaningful regression checks for metric aggregation and checkpoint selection."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from dpp_evaluation import (Artifact, SemanticMetricsAccumulator, confusion_matrix,
                           instance_statistics, size_metrics, lighting_summary, remap_gt,
                           evaluation_signature, SOURCE_FILES)


class EvaluationTests(unittest.TestCase):
    def test_semantic_mapping_and_ignore(self):
        gt=remap_gt(np.array([[0,1,2],[3,4,255]],dtype=np.uint16))
        np.testing.assert_array_equal(gt,[[0,1,2],[1,2,255]])
        pred=np.array([[0,2,2],[1,0,2]],dtype=np.uint8)
        np.testing.assert_array_equal(confusion_matrix(gt,pred),[[1,0,0],[0,1,1],[1,0,1]])
        with self.assertRaises(ValueError):remap_gt(np.array([[7]]))

    def test_global_lighting_uses_pooled_cm(self):
        acc=SemanticMetricsAccumulator()
        acc.update(np.array([[2]]),np.array([[2]]),'one','Sunny I')
        acc.update(np.full((1,9),2),np.zeros((1,9),dtype=int),'two','Sunny I')
        self.assertAlmostEqual(acc.compute_global_metrics()['weed_iou'],10)
        self.assertAlmostEqual(lighting_summary(acc.per_image_records).iloc[0].weed_iou,10)
        self.assertNotAlmostEqual(np.mean([r['weed_iou'] for r in acc.per_image_records]),10)

    def test_gt_instance_ids_not_connected_components(self):
        # Two touching plants remain two GT instances, not one connected component.
        gt=np.array([[2,2,2,2],[0,255,1,0]])
        ids=np.array([[1,1,2,2],[0,0,0,0]])
        pred=np.array([[2,2,0,2],[2,2,0,0]])
        bins=instance_statistics(gt,pred,ids);q=bins['Q1 (Micro)']
        self.assertEqual(q,{'instances':2,'gt_pixels':4,'tp':3,'fp_shared_nonweed':1,'fn':1,'hits_05':2})
        metrics=size_metrics(q)
        self.assertEqual(metrics['weed_iou'],60)
        self.assertEqual(metrics['weed_prec'],75)
        self.assertEqual(metrics['weed_rec'],75)
        self.assertEqual(metrics['hr_05'],100)
        self.assertEqual(bins['Q4 (Large)']['fp_shared_nonweed'],1)

    def test_empty_weed_and_missing_ids(self):
        z=np.zeros((3,3),dtype=np.uint8)
        bins=instance_statistics(z,z,z)
        self.assertTrue(all(v['instances']==0 for v in bins.values()))
        self.assertIsNone(size_metrics(bins['Q1 (Micro)'])['hr_05'])
        with self.assertRaises(ValueError):instance_statistics(np.full((2,2),2),np.zeros((2,2)),np.zeros((2,2)))

    def test_checkpoint_selection_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'artifacts.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('training_output/checkpoints/last.ckpt',b'not a selected checkpoint')
            artifact=Artifact(p)
            with self.assertRaises(ValueError):artifact.checkpoint('best_mIoU')
            with zipfile.ZipFile(p,'a') as z:
                z.writestr('training_output/checkpoints/one_best_mIoU_epoch1.ckpt',b'x')
                z.writestr('training_output/checkpoints/two_best_mIoU_epoch2.ckpt',b'y')
            with self.assertRaises(ValueError):Artifact(p).checkpoint('best_mIoU')

    def test_cache_identity_tracks_factory_and_effective_backbone_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in SOURCE_FILES:
                p=root/'RPD_P2'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source v1')
            entry={'fingerprint':'weights','config':{'backbone':{'downsampling':{'epsilon_squared':.001}}}}
            first=evaluation_signature(root,entry,'dataset','cpu','full')
            (root/'RPD_P2/models/rpdnet/downsampling.py').write_text('source v2')
            second=evaluation_signature(root,entry,'dataset','cpu','full')
            self.assertNotEqual(first,second)
            entry['config']['backbone']['downsampling']['epsilon_squared']=.01
            self.assertNotEqual(second,evaluation_signature(root,entry,'dataset','cpu','full'))


if __name__=='__main__':unittest.main()
