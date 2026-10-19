"""DeepDown integration: Full AAA + training-only auxiliary supervision at x6.

One campaign candidate (ex0) and the unchanged P2 WDice control.
"""
import argparse
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
import yaml
from models import get_backbone, get_criterion
from models.model_multimetrics import SegmentationNetwork
from models.rpdnet.aaa import AAAMaxBlurPool, tri3_filter
from scripts.export_deepsup import export, load_training_model
from test_deepsup import trainer_smoke, pipeline_smoke

ROOT = Path(__file__).resolve().parent
SCENARIOS = {'control': 'DeepDown_Control', 'ex0': 'DeepDown_Ex0'}


def config(mode):
    return yaml.safe_load((ROOT/'configs/p35_deepdown'/f'{SCENARIOS[mode]}.yaml').read_text())


def module(cfg):
    return SegmentationNetwork(get_backbone(cfg), get_criterion(cfg),
        cfg['train']['learning_rate'], cfg['train']['weight_decay'],
        train_step_settings=['regular'], val_step_settings=['regular'],
        scheduler_type=cfg['train']['scheduler'], warmup_epochs=cfg['train']['warmup_epochs'])


def smoke(cfg, output, accelerator='cpu', devices=1, resume=False):
    def factory(c):
        model = get_backbone(c)
        # Deliberately corrupt nonzero-rank initialization: DDP must broadcast
        # the original persistent AAA buffers from rank0 before the first step.
        if int(os.environ.get('LOCAL_RANK', '0')) == 1 and not resume:
            for pool in model.modules():
                if isinstance(pool, AAAMaxBlurPool): pool.filt.zero_()
        return model
    with patch('test_deepsup.get_backbone', side_effect=factory):
        trainer_smoke(cfg, output, accelerator, devices, resume)
    trained = load_training_model(cfg, Path(output)/'checkpoints/last.ckpt')
    pools = [m for m in trained.modules() if isinstance(m, AAAMaxBlurPool)]
    assert len(pools) == (4 if cfg['backbone'].get('downsampling') else 0)
    for pool in pools:
        assert torch.equal(pool.filt, tri3_filter(pool.channels))


class DeepDownTests(unittest.TestCase):
    def test_exact_parent_recipes_and_no_new_model_hyperparameters(self):
        base = yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        aaa = yaml.safe_load((ROOT/'configs/p3_scenarios/P3_AAA_Ex3_All.yaml').read_text())
        deep = yaml.safe_load((ROOT/'configs/p3_deepsup/P3_DeepSup_Ex4_x6.yaml').read_text())
        for mode in SCENARIOS:
            cfg = config(mode); recipe = deepcopy(cfg)
            recipe['experiment'] = base['experiment']
            down = recipe['backbone'].pop('downsampling', None)
            aux = recipe['backbone'].pop('deep_supervision', None)
            self.assertEqual(recipe, base)
            if mode == 'ex0':
                self.assertEqual(cfg['experiment']['id'], 'DeepDown-Ex0')
                self.assertEqual(down, aaa['backbone']['downsampling'])
                self.assertEqual(aux, deep['backbone']['deep_supervision'])
                self.assertEqual(aux['tap'], 'x6')
            else:
                self.assertIsNone(down); self.assertIsNone(aux)

    def test_fixed_aaa_common_initialization_rng_and_main_identity(self):
        cfg = config('ex0'); aaa_cfg = deepcopy(cfg); aaa_cfg['backbone'].pop('deep_supervision')
        torch.manual_seed(cfg['seed']); baseline = get_backbone(config('control'))
        common = {k: v.clone() for k, v in baseline.state_dict().items()}
        rng = torch.get_rng_state().clone()
        torch.manual_seed(cfg['seed']); combined = get_backbone(cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in common.items(): self.assertTrue(torch.equal(value, combined.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in combined.parameters()), 208422)
        for stage, channels in enumerate([16, 32, 64, 128], 1):
            pool = getattr(combined, f'down{stage}')[0]
            self.assertIsInstance(pool, AAAMaxBlurPool)
            self.assertEqual(sum(p.numel() for p in pool.parameters()), 0)
            self.assertTrue(torch.equal(pool.filt, tri3_filter(channels)))
        torch.manual_seed(cfg['seed']); aaa_only = get_backbone(aaa_cfg)
        combined.train(); aaa_only.train(); x = torch.randn(2, 3, 33, 35)
        with torch.no_grad():
            torch.testing.assert_close(combined(x, return_aux=True)['main'], aaa_only(x), rtol=0, atol=0)
        self.assertEqual(sum(isinstance(getattr(baseline, f'down{i}')[0], nn.MaxPool2d) for i in range(1, 5)), 4)

    def test_x6_tap_single_objective_and_aux_upstream_gradient(self):
        cfg = config('ex0'); net = module(cfg).train(); seen = {}
        h1 = net.network.up1.register_forward_hook(lambda m, i, o: seen.update(up1=o))
        h2 = net.network.aux_head.register_forward_pre_hook(lambda m, i: seen.update(aux_input=i[0]))
        batch = {'input_image': torch.randn(2, 3, 33, 35),
                 'anno': torch.randint(0, 3, (2, 33, 35))}
        captured = {}
        h3 = net.network.register_forward_hook(lambda m, i, o: captured.update(o))
        net._trainer = SimpleNamespace(current_epoch=0)
        result = net.training_step(batch, 0)
        out = captured
        h1.remove(); h2.remove(); h3.remove()
        self.assertIs(seen['up1'], seen['aux_input']); self.assertEqual(seen['up1'].shape[1], 64)
        main = net.criterion(out['main'], batch['anno'], mode='train', mask_keep=batch['anno'] != 255)
        aux = net.criterion(out['aux'], batch['anno'], mode='train', mask_keep=batch['anno'] != 255)
        torch.testing.assert_close(result['loss'], main + .4*aux, rtol=0, atol=0)
        aux.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.network.in_conv.parameters()))
        for name in ['up2', 'up3', 'up4', 'out_conv']:
            self.assertTrue(all(p.grad is None for p in getattr(net.network, name).parameters()))
        optimizers, _ = net.configure_optimizers()
        groups = optimizers[0].param_groups
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]['initial_lr'], cfg['train']['learning_rate'])
        self.assertEqual(optimizers[0].defaults['lr'], cfg['train']['learning_rate'])
        self.assertEqual(groups[0]['weight_decay'], cfg['train']['weight_decay'])
        self.assertEqual({id(p) for p in groups[0]['params']}, {id(p) for p in net.parameters()})

    def test_checkpoint_guards_both_components_and_main_export_keeps_aaa(self):
        cfg = config('ex0'); net = module(cfg)
        payload = {'state_dict': net.state_dict()}; net.on_save_checkpoint(payload)
        self.assertEqual(payload['downsampling'], cfg['backbone']['downsampling'])
        self.assertEqual(payload['deep_supervision']['tap'], 'x6')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'combined.ckpt'; torch.save(payload, path)
            load_training_model(cfg, path)
            wrong = deepcopy(cfg); wrong['backbone']['downsampling']['stages'] = [1, 2]
            with self.assertRaisesRegex(ValueError, 'downsampling'): module(wrong).on_load_checkpoint(payload)
            with self.assertRaisesRegex(ValueError, 'downsampling'): load_training_model(wrong, path)
            report = export(cfg, path, Path(tmp)/'main', sizes=((32, 32), (33, 35)))
            self.assertEqual(report['downsampling_preserved'], cfg['backbone']['downsampling'])
            self.assertEqual(report['main_trainable_params'], 189827)
            bare_cfg = yaml.safe_load((Path(tmp)/'main/main_config.yaml').read_text())
            self.assertNotIn('deep_supervision', bare_cfg['backbone'])
            self.assertEqual(bare_cfg['backbone']['downsampling'], cfg['backbone']['downsampling'])
            bare = get_backbone(bare_cfg).eval()
            bare.load_state_dict(torch.load(Path(tmp)/'main/main_state_dict.pt', weights_only=True), strict=True)
            self.assertEqual(sum(isinstance(m, AAAMaxBlurPool) for m in bare.modules()), 4)
            self.assertFalse(any('aux_head' in k for k in bare.state_dict()))

    def test_notebook_source_modes_and_new_ancestry_contract(self):
        nb = json.loads((ROOT/'notebook/kaggle_train_repdwnet_deepdown.ipynb').read_text())
        first = next(c for c in nb['cells'] if c['cell_type'] == 'code'); ns = {}; exec(''.join(first['source']), ns)
        self.assertEqual(ns['GIT_REF'], 'f/deepdown'); self.assertEqual(ns['EXPERIMENT_MODE'], 'ex0')
        self.assertEqual(ns['DEVICES'], 2); self.assertEqual(set(ns['EXPERIMENTS']), {'control', 'ex0'})
        text = '\n'.join(''.join(c['source']) for c in nb['cells'])
        for token in ['3100c6fd1f2bb9fe3887f2b15a0a306054e21c38',
                      '0d2f5c55299a297feae3aecdbe9d2a8ae276e93d',
                      'test_deepdown.py', 'test_validation_infrastructure.py', 'campaign_audit.json']:
            self.assertIn(token, text)
        for c in nb['cells']:
            if c['cell_type'] == 'code':
                ast.parse(''.join(l for l in c['source'] if not l.lstrip().startswith('!')))
                self.assertIsNone(c['execution_count']); self.assertEqual(c['outputs'], [])


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--pipeline-smoke', action='store_true')
    parser.add_argument('--native-smoke', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--config'); parser.add_argument('--output-dir')
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='cpu'); parser.add_argument('--devices', type=int, default=1)
    args, rest = parser.parse_known_args()
    if args.smoke or args.pipeline_smoke:
        if not args.config or not args.output_dir: parser.error('config and output-dir required')
        cfg = yaml.safe_load(Path(args.config).read_text())
        if args.smoke: smoke(cfg, args.output_dir, args.accelerator, args.devices, args.resume)
        else: pipeline_smoke(cfg, args.output_dir, args.accelerator, args.devices)
    elif args.native_smoke:
        device = torch.device('cuda' if args.accelerator == 'gpu' else 'cpu')
        if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('Native GPU smoke requires CUDA')
        records = []
        for key in SCENARIOS:
            model = get_backbone(config(key)).to(device).train()
            with torch.no_grad():
                for size in [768, 1024]:
                    x = torch.randn(1, 3, size, size, device=device)
                    out = model(x, return_aux=True) if key == 'ex0' else {'main': model(x)}
                    for name, value in out.items():
                        assert value.shape == (1, 3, size, size) and torch.isfinite(value).all()
                    records.append({'mode': key, 'size': size, 'status': 'passed'})
            del model
            if device.type == 'cuda': torch.cuda.empty_cache()
        if args.output_dir:
            out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
            (out/'native_shapes.json').write_text(json.dumps(records, indent=2))
        print('PASS DeepDown native shapes control/ex0, 768/1024', flush=True)
    else:
        unittest.main(argv=[__file__, *rest])
