"""DeDo integration: inherited recipes, x6 gradients, A-oA widths and strict exports."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
import yaml
from torch import nn
from test_deepdown import ROOT, module
from models import get_backbone, get_criterion
from models.rpdnet.dpp import LiteDPPSym
from models.rpdnet.lip import BottleneckLIP
from models.rpdnet.aaa import AAAMaxBlurPool
from models.rpdnet.fuse_fpn import AddUp
from scripts.export_deepsup import export, load_training_model
from utils.deepdown_campaign import EXPERIMENTS


def config(mode):
    return yaml.safe_load((ROOT/'configs/p35_deepdown'/f'{EXPERIMENTS[mode][1]}.yaml').read_text())


class DeDoTests(unittest.TestCase):
    def test_inherited_recipe_and_exact_parent_components(self):
        base = yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        parents = {'3Aall':'P3_AAA_Ex3_All', 'LipEar':'P3_LIP_Ex1_Early', 'DppAll':'P3_DPP_Ex3_All'}
        for mode in EXPERIMENTS:
            with self.subTest(mode=mode):
                cfg = config(mode); recipe = deepcopy(cfg)
                down = recipe['backbone'].pop('downsampling', None)
                aux = recipe['backbone'].pop('deep_supervision', None)
                fusion = recipe['backbone'].pop('fusion', None)
                spatial = recipe['backbone'].pop('spatial_block', 'repdw')
                self.assertEqual(spatial, 'pdc' if mode.endswith('.pdc') else 'repdw')
                recipe['experiment'] = base['experiment']
                self.assertEqual(recipe, base)
                self.assertEqual(fusion, {'variant':'add'} if '.elwAdd' in mode else None)
                if mode != 'control':
                    key = '3Aall' if mode == 'ex0' else mode.split('.')[1]
                    parent = yaml.safe_load((ROOT/'configs/p3_scenarios'/f'{parents[key]}.yaml').read_text())
                    self.assertEqual(down, parent['backbone']['downsampling'])
                    self.assertEqual(aux, config('ex0')['backbone']['deep_supervision'])

    def test_forward_x6_loss_gradients_and_optimizer_all_candidates(self):
        for mode in list(EXPERIMENTS)[2:]:
            with self.subTest(mode=mode):
                cfg = config(mode); torch.manual_seed(cfg['seed']); net = module(cfg).train()
                backbone = net.network; seen = {}
                h = backbone.aux_head.register_forward_pre_hook(lambda m, x: seen.update(tap=x[0]))
                x = torch.randn(2,3,33,35); target = torch.randint(0,3,(2,33,35))
                out = backbone(x, return_aux=True); h.remove()
                self.assertEqual(seen['tap'].shape, (2,64,4,4))
                criterion = get_criterion(cfg)
                loss = sum(w*criterion(out[k], target, mode='train', mask_keep=target!=255)
                           for k,w in [('main',1),('aux',.4)])
                self.assertTrue(torch.isfinite(loss)); loss.backward()
                for part in [backbone.in_conv, backbone.up1, backbone.aux_head, backbone.out_conv]:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in part.parameters()))
                for p in backbone.parameters():
                    if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
                net._trainer = SimpleNamespace(current_epoch=0, max_epochs=100)
                optimizers,_ = net.configure_optimizers(); opt = optimizers[0]
                groups = opt.param_groups
                self.assertEqual({id(p) for g in groups for p in g['params']}, {id(p) for p in net.parameters()})
                dpp = [m for m in backbone.modules() if isinstance(m,LiteDPPSym)]
                lip = [m for m in backbone.modules() if isinstance(m,BottleneckLIP)]
                aaa = [m for m in backbone.modules() if isinstance(m,AAAMaxBlurPool)]
                self.assertEqual((len(dpp),len(lip),len(aaa)), (4,0,0) if 'DppAll' in mode else ((0,2,0) if 'LipEar' in mode else (0,0,4)))
                if dpp:
                    for pool in dpp:
                        for p in pool.parameters():
                            self.assertTrue(torch.isfinite(p.grad).all())
                            group = next(g for g in groups if any(q is p for q in g['params']))
                            self.assertEqual(group['weight_decay'],0)
                opt.step()
                if '.elwAdd' in mode:
                    for up,c,o in zip([backbone.up1,backbone.up2,backbone.up3,backbone.up4],[128,64,32,16],[64,32,16,16]):
                        self.assertIsInstance(up,AddUp); self.assertIsInstance(up.lateral,nn.Identity)
                        self.assertEqual((up.conv.in_channels,up.conv.mid_channels,up.conv.out_channels),(c,c,o))
                else:
                    self.assertFalse(any(isinstance(m,AddUp) for m in backbone.modules()))

    def test_strict_resume_and_main_export_preserve_components(self):
        for mode in list(EXPERIMENTS)[2:]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg = config(mode); net = module(cfg)
                payload = {'state_dict':net.state_dict()}; net.on_save_checkpoint(payload)
                path = Path(tmp)/'model.ckpt'; torch.save(payload,path)
                net.on_load_checkpoint(payload)
                report = export(cfg,path,Path(tmp)/'main',sizes=((32,32),(33,35)))
                self.assertEqual(report['downsampling_preserved'],cfg['backbone']['downsampling'])
                self.assertEqual(report['feature_fusion_preserved'],cfg['backbone'].get('fusion'))
                exported_cfg = yaml.safe_load((Path(tmp)/'main/main_config.yaml').read_text())
                bare = get_backbone(exported_cfg).eval()
                bare.load_state_dict(torch.load(Path(tmp)/'main/main_state_dict.pt',weights_only=True),strict=True)
                self.assertIsNone(bare.aux_head)
                self.assertEqual(bare.fusion,cfg['backbone'].get('fusion'))
                wrong = deepcopy(cfg)
                if 'fusion' in wrong['backbone']:wrong['backbone'].pop('fusion')
                else:wrong['backbone']['fusion']={'variant':'add'}
                with self.assertRaisesRegex(ValueError,'fusion'):load_training_model(wrong,path)
                with self.assertRaisesRegex(ValueError,'fusion'):module(wrong).on_load_checkpoint(payload)

    def test_invalid_fusion_fails_before_training(self):
        cfg = config('ex0'); cfg['backbone']['fusion']={'variant':'fpn'}
        with self.assertRaises(ValueError):get_backbone(cfg)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
