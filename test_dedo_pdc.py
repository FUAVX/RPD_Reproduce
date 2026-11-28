"""B0-equivalent PDC restoration, PW sharing and spatial checkpoint safety."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import torch
from torch import nn
from torch.nn import functional as F
import yaml
from test_dedo import config
from test_deepdown import module, ROOT
from models import get_backbone
from models.rpdnet.RPDNet import RPDNet
from models.rpdnet.RPD_Module import RPD
from models.rpdnet.rep_dw import RepDW
from models.rpdnet.aaa import AAAMaxBlurPool
from scripts.export_deepsup import export, load_training_model

MODES = ['alx6.3Aall.concat.pdc', 'alx6.3Aall.elwAdd.pdc']


class PDCRestorationTests(unittest.TestCase):
    def test_exact_parent_recipes(self):
        for mode, parent in zip(MODES, ['ex0', 'alx6.3Aall.elwAdd']):
            child = config(mode); original = config(parent)
            self.assertEqual(child['backbone'].pop('spatial_block'), 'pdc')
            self.assertEqual(child['experiment']['id'], 'DeDo-'+mode)
            child['experiment'] = original['experiment']
            self.assertEqual(child, original)

    def test_pure_pdc_host_exactly_matches_b0_initialization_output_and_gradients(self):
        cfg = config(MODES[0]); cfg['backbone'].pop('deep_supervision'); cfg['backbone'].pop('downsampling')
        torch.manual_seed(cfg['seed']); expected = RPDNet(num_classes=3).train()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(cfg['seed']); actual = get_backbone(cfg).train()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        for k, v in expected.state_dict().items():
            self.assertTrue(torch.equal(v, actual.state_dict()[k]), k)
        self.assertEqual(sum(p.numel() for p in actual.parameters()), 189827)
        x = torch.randn(2,3,33,35)
        a = expected(x); b = actual(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        for (ka,pa),(kb,pb) in zip(expected.named_parameters(),actual.named_parameters()):
            self.assertEqual(ka,kb)
            torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_six_dw_paths_and_shared_pw_paths_in_every_stage(self):
        for mode in MODES:
            model = get_backbone(config(mode))
            self.assertFalse(any(isinstance(m,RepDW) for m in model.modules()))
            spatial = [m for m in model.modules() if isinstance(m,RPD) and m.kernel_size==3]
            pw = [m for m in model.modules() if isinstance(m,RPD) and m.kernel_size==1]
            self.assertEqual(len(spatial),16); self.assertEqual(len(pw),16)
            self.assertEqual(sum(isinstance(m,AAAMaxBlurPool) for m in model.modules()),4)
            for m in spatial:
                self.assertEqual(len(m.rbr_conv),4)
                self.assertIs(m.pdc[0],F.conv2d)
                self.assertEqual([op.op_type for op in m.pdc[1:]],['cd','ad','rd'])
                self.assertEqual([op.theta for op in m.pdc[1:]],[.5,.5,.5])
                self.assertEqual(len({id(branch) for branch in m.rbr_conv}),4)
                self.assertIsInstance(m.rbr_identity,nn.BatchNorm2d)
                self.assertIsNotNone(m.rbr_1x1)
                self.assertEqual(m.groups,m.in_channels)
                self.assertEqual(m.rbr_1x1.conv.groups,m.in_channels)
                self.assertIsInstance(m.se,nn.Identity)
            for m in pw:
                self.assertEqual(len(m.rbr_conv),4)
                self.assertEqual(len({id(branch) for branch in m.rbr_conv}),1)
                self.assertIsNone(m.rbr_1x1)
                self.assertEqual(m.groups,1)
                same = m.rbr_conv[0].conv.in_channels == m.rbr_conv[0].conv.out_channels
                self.assertEqual(isinstance(m.rbr_identity,nn.BatchNorm2d),same)
            # The spatial sum and following ReLU follow the original RPD unit.
            unit=spatial[0].eval();x=torch.randn(2,unit.in_channels,17,19)
            with torch.no_grad():
                expected=unit.rbr_identity(x)+unit.rbr_1x1(x)
                for branch in unit.rbr_conv:expected=expected+branch(x)
                torch.testing.assert_close(unit(x),F.relu(expected),atol=0,rtol=0)

    def test_pdc_checkpoint_cannot_be_confused_with_same_shape_repdw(self):
        for mode,parent in zip(MODES,['ex0','alx6.3Aall.elwAdd']):
            cfg=config(mode); net=module(cfg); old=module(config(parent))
            # Current prefixes differ (rbr_conv vs rbr_dw), while the underlying
            # tensor geometry can match; explicit metadata also protects renamed weights.
            renamed={k.replace('.rbr_dw.', '.rbr_conv.'):v for k,v in old.state_dict().items()}
            self.assertEqual(set(net.state_dict()),set(renamed))
            for k,v in net.state_dict().items():self.assertEqual(v.shape,renamed[k].shape)
            payload={'state_dict':net.state_dict()};net.on_save_checkpoint(payload)
            self.assertEqual(payload['spatial_block'],'pdc')
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'pdc.ckpt';torch.save(payload,path)
                load_training_model(cfg,path)
                with self.assertRaisesRegex(ValueError,'spatial block'):old.on_load_checkpoint(payload)
                with self.assertRaisesRegex(ValueError,'spatial block'):load_training_model(config(parent),path)
                missing=deepcopy(payload);missing.pop('spatial_block')
                with self.assertRaisesRegex(ValueError,'spatial block'):net.on_load_checkpoint(missing)
                report=export(cfg,path,Path(tmp)/'main',sizes=((32,32),(33,35)))
                self.assertEqual(report['spatial_block_preserved'],'pdc')
                main_cfg=yaml.safe_load((Path(tmp)/'main/main_config.yaml').read_text())
                self.assertEqual(main_cfg['backbone']['spatial_block'],'pdc')
                self.assertNotIn('deep_supervision',main_cfg['backbone'])
                main=get_backbone(main_cfg).eval()
                main.load_state_dict(torch.load(Path(tmp)/'main/main_state_dict.pt',weights_only=True),strict=True)
                self.assertEqual(main.spatial_block,'pdc')
                self.assertIsNone(main.aux_head)
                with self.assertRaisesRegex(ValueError,'two-step'):export(cfg,path,Path(tmp)/'fused',fuse=True)
            # Legacy RepDW checkpoints without the new field remain loadable.
            legacy={'state_dict':old.state_dict()};old.on_save_checkpoint(legacy);legacy.pop('spatial_block')
            old.on_load_checkpoint(legacy)
            with self.assertRaisesRegex(ValueError,'spatial block'):net.on_load_checkpoint(legacy)

    def test_invalid_spatial_selection_and_direct_pdc_deploy_rejected(self):
        for key,value in [('spatial_block','unknown'),('deploy',True),('base_c',32)]:
            cfg=config(MODES[0]);cfg['backbone'][key]=value
            with self.assertRaises(ValueError):get_backbone(cfg)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
