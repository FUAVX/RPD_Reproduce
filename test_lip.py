"""LIP operator/integration tests and actual trainer DDP smoke runner.

python test_lip.py
python test_lip.py --smoke --config configs/p3_scenarios/P3_LIP_Ex1_Early.yaml \
  --output-dir RUN/smoke_early --accelerator gpu --devices 2
"""
import argparse
import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

from models import get_backbone, get_criterion
from models.model_multimetrics import SegmentationNetwork
from models.rpdnet.lip import BottleneckLIP, lip2d
from models.rpdnet.downsampling import make_downsampling
from models.rpdnet.dpp import dpp_optimizer_groups
from models.rpdnet.RPD_Module import RPD_model_deploy
from scripts.analyze_lip import load_checkpoint, window_view

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml'
SCENARIOS = [(1, 'Early', [1, 2], 1504), (2, 'Late', [3, 4], 22336),
             (3, 'Middle', [2, 3], 5728), (4, 'First3', [1, 2, 3], 6048),
             (5, 'All', [1, 2, 3, 4], 23840)]


class SyntheticSegmentation(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {'input_image': torch.randn(3, 32, 32, generator=generator),
                'anno': torch.randint(0, 3, (32, 32), generator=generator)}


def trainer_smoke(config_path, output_dir, accelerator='cpu', devices=1, resume=False):
    """Uses actual P2 Lightning module, WDice, config callback, logger and checkpoint."""
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    from callbacks.config_callback import ConfigCallback
    cfg = yaml.safe_load(Path(config_path).read_text())
    pl.seed_everything(cfg['seed'])
    module = SegmentationNetwork(get_backbone(cfg), get_criterion(cfg),
                                 cfg['train']['learning_rate'], cfg['train']['weight_decay'],
                                 train_step_settings=['regular'], val_step_settings=['regular'],
                                 scheduler_type=cfg['train']['scheduler'],
                                 warmup_epochs=cfg['train']['warmup_epochs'])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(dirpath=str(output / 'checkpoints'), save_last=True,
                                 monitor='val_weed_iou', mode='max')
    logger = TensorBoardLogger(str(output), name='lightning_logs', version=0, default_hp_metric=False)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
                         strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=90))
                         if devices > 1 else 'auto',
                         precision=32, max_epochs=2 if resume else 1,
                         callbacks=[ConfigCallback(cfg), checkpoint], logger=logger,
                         default_root_dir=str(output), check_val_every_n_epoch=1,
                         limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
                         enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1)
    loader = DataLoader(SyntheticSegmentation(), batch_size=2)
    trainer.fit(module, loader, loader,
                ckpt_path=str(output / 'checkpoints/last.ckpt') if resume else None)
    if trainer.is_global_zero:
        path = output / 'checkpoints/last.ckpt'
        raw = torch.load(path, map_location='cpu', weights_only=False)
        assert any('logit.conv3.weight' in key for key in raw['state_dict'])
        assert raw['optimizer_states'] and raw['global_step'] > 0
        assert all(torch.isfinite(value).all() for value in raw['state_dict'].values())
        assert any(torch.count_nonzero(value) > 0 for key, value in raw['state_dict'].items()
                   if 'logit.conv3.weight' in key)
        fresh = get_backbone(cfg).eval()
        load_checkpoint(fresh, path)
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(logger.log_dir).Reload()
        assert events.Scalars('val_weed_iou')
        print(f'PASS trainer smoke: {Path(config_path).stem}, devices={devices}, resume={resume}')


class LIPTests(unittest.TestCase):
    def test_operator_reference_and_gradients(self):
        for h, w in [(8, 10), (7, 9), (8, 9)]:
            x = torch.randn(2, 3, h, w, dtype=torch.double, requires_grad=True)
            z = torch.randn_like(x, requires_grad=True)
            expected = (window_view(x) * window_view(z).softmax(-1)).sum(-1)
            y = lip2d(x, z)
            torch.testing.assert_close(y, expected)
            actual_grad = torch.autograd.grad(y.square().sum(), (x, z), retain_graph=True)
            ref_grad = torch.autograd.grad(expected.square().sum(), (x, z))
            for actual, ref in zip(actual_grad, ref_grad):
                torch.testing.assert_close(actual, ref)
        x = torch.randn(1, 1, 4, 4, dtype=torch.double, requires_grad=True)
        z = torch.randn_like(x, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lip2d, (x, z)))

    def test_module_init_params_and_learning(self):
        for c, params in [(16, 320), (32, 1184), (64, 4544), (128, 17792)]:
            m = BottleneckLIP(c)
            self.assertEqual(sum(p.numel() for p in m.parameters()), params)
            for layer in m.modules():
                if isinstance(layer, nn.InstanceNorm2d):
                    self.assertEqual(layer.eps, 1e-5)
                    self.assertFalse(layer.track_running_stats)
                    self.assertTrue(torch.equal(layer.weight, torch.ones_like(layer.weight)))
                    self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
                if isinstance(layer, nn.Conv2d):
                    self.assertIsNone(layer.bias)
            x = torch.randn(2, c, 7, 9)
            torch.testing.assert_close(m(x), F.avg_pool2d(x, 2, 2))
            torch.testing.assert_close(m.logit(x), torch.full_like(x, 6))
        m = BottleneckLIP(16)
        x = torch.randn(2, 16, 8, 8)
        optimizer = torch.optim.Adam(dpp_optimizer_groups(m, 2e-4, 2e-4))
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 2e-4)
        ids = [id(p) for g in optimizer.param_groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in m.parameters()})
        for _ in range(2):
            optimizer.zero_grad()
            m(x).square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            optimizer.step()
        self.assertGreater(m.logit.conv1.weight.grad.abs().max().item(), 0)

    def test_backbone_initialization_and_rng_match_control(self):
        baseline = yaml.safe_load(BASE.read_text())
        seed = baseline['seed']
        torch.manual_seed(seed)
        control = get_backbone(baseline)
        expected_state = control.state_dict()
        expected_rng = torch.get_rng_state().clone()
        expected_next_draw = torch.randn(8)
        shared_lip_states = {}
        for ex, name, stages, _ in SCENARIOS:
            cfg = yaml.safe_load((ROOT / f'configs/p3_scenarios/P3_LIP_Ex{ex}_{name}.yaml').read_text())
            torch.manual_seed(seed)
            candidate = get_backbone(cfg)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng), name)
            self.assertTrue(torch.equal(torch.randn(8), expected_next_draw), name)
            actual_state = candidate.state_dict()
            for key, value in expected_state.items():
                self.assertIn(key, actual_state)
                self.assertTrue(torch.equal(actual_state[key], value), f'{name}: {key}')
            for stage in stages:
                pool = getattr(candidate, f'down{stage}')[0]
                # A shared stage starts from the same G state across placements.
                if stage in shared_lip_states:
                    for key, value in pool.state_dict().items():
                        self.assertTrue(torch.equal(value, shared_lip_states[stage][key]))
                else:
                    shared_lip_states[stage] = copy.deepcopy(pool.state_dict())
                self.assertTrue(torch.count_nonzero(pool.logit.conv1.weight) > 0)
                self.assertEqual(torch.count_nonzero(pool.logit.conv3.weight).item(), 0)
        torch.manual_seed(seed + 1)  # Test only: G remains responsive to the user's seed.
        different_seed = get_backbone(cfg)
        stage = stages[0]
        self.assertFalse(torch.equal(getattr(different_seed, f'down{stage}')[0].logit.conv1.weight,
                                     shared_lip_states[stage]['logit.conv1.weight']))

    def test_configs_full_model_checkpoint_and_deploy(self):
        baseline = yaml.safe_load(BASE.read_text())
        torch.manual_seed(1)
        control = get_backbone(baseline).eval()
        explicit = copy.deepcopy(baseline)
        explicit['backbone']['downsampling'] = {'variant': 'maxpool', 'stages': []}
        torch.manual_seed(1)
        unchanged = get_backbone(explicit).eval()
        for key, value in control.state_dict().items():
            self.assertTrue(torch.equal(value, unchanged.state_dict()[key]))
        x = torch.randn(1, 3, 33, 35)
        for ex, name, stages, added in SCENARIOS:
            cfg = yaml.safe_load((ROOT / f'configs/p3_scenarios/P3_LIP_Ex{ex}_{name}.yaml').read_text())
            inherited = copy.deepcopy(cfg)
            del inherited['backbone']['downsampling']
            inherited['experiment'] = baseline['experiment']
            self.assertEqual(inherited, baseline)
            model = get_backbone(cfg).eval()
            self.assertEqual([i for i in range(1, 5) if isinstance(getattr(model, f'down{i}')[0], BottleneckLIP)], stages)
            self.assertEqual(sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters()), added)
            # Bound untrained multi-branch activations to meaningfully check fusion tolerance.
            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.mul_(.1)
                    elif isinstance(layer, nn.BatchNorm2d):
                        layer.running_mean.uniform_(-.1, .1)
                        layer.running_var.uniform_(.7, 1.3)
                y = model(x)
            self.assertEqual(tuple(y.shape), (1, 3, 33, 35))
            restored = get_backbone(cfg).eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            deployed = RPD_model_deploy(model, do_copy=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(restored(x), y, rtol=0, atol=0)
                torch.testing.assert_close(deployed(x), y, rtol=1e-4, atol=1e-4)
            self.assertEqual(sum(isinstance(m, BottleneckLIP) for m in deployed.modules()), len(stages))

    def test_config_rejections(self):
        for cfg in [dict(variant='lip_bottleneck', stages=stages) for stages in
                    [[], [1], [1, 1], [1, 5], [1, 2, 4], [1, 3], [True, 2], [1, 2, 3, 3], [4, 3, 2, 1]]]:
            with self.assertRaises(ValueError):
                make_downsampling(16, cfg, 1)
        with self.assertRaises(ValueError):
            make_downsampling(16, {'variant': 'lip_bottleneck', 'stages': [1, 2], 'temperature': 1}, 1)

    def test_notebook_modes_and_launch_contract(self):
        import ast
        import json
        notebook = json.loads((ROOT / 'notebook/kaggle_train_repdwnet_lip.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(line for line in cell['source'] if not line.startswith('!')))
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        namespace = {}
        exec(''.join(notebook['cells'][1]['source']), namespace)
        self.assertEqual(namespace['GIT_REF'], 'f/down-lip')
        self.assertEqual(namespace['DEVICES'], 2)
        self.assertEqual(namespace['EXPERIMENTS']['ex3'], ('P3_LIP_Ex3_Middle', [2, 3]))
        self.assertEqual(namespace['EXPERIMENTS']['ex4'], ('P3_LIP_Ex4_First3', [1, 2, 3]))
        self.assertEqual(namespace['EXPERIMENTS']['ex5'], ('P3_LIP_Ex5_All', [1, 2, 3, 4]))
        for _, (scenario, stages) in namespace['EXPERIMENTS'].items():
            cfg = yaml.safe_load((ROOT / 'configs/p3_scenarios' / (scenario+'.yaml')).read_text())
            self.assertEqual(cfg['backbone']['downsampling']['stages'], stages)
        launch = ''.join(notebook['cells'][5]['source'])
        self.assertIn("'--devices', str(DEVICES)", launch)
        self.assertIn('scripts/benchmark_lip.py', launch)
        self.assertIn("'--lip-config', str(scenario_path)", launch)
        trainer_source = (ROOT / 'multi_metric_train.py').read_text()
        self.assertIn("target_max_epochs=cfg['train']['max_epoch']", trainer_source)
        self.assertIn("val_interval=cfg['val']['check_val_every_n_epoch']", trainer_source)

    def test_actual_trainer_and_resume(self):
        for scenario in ['P3_LIP_Ex1_Early', 'P3_LIP_Ex4_First3', 'P3_LIP_Ex5_All']:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as output:
                cfg = ROOT / 'configs/p3_scenarios' / (scenario + '.yaml')
                trainer_smoke(cfg, output)
                trainer_smoke(cfg, output, resume=True)
                state = torch.load(Path(output) / 'checkpoints/last.ckpt', weights_only=False)
                self.assertEqual(state['epoch'], 1)
                self.assertEqual(state['global_step'], 4)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--output-dir')
    parser.add_argument('--accelerator', default='cpu')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--resume', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.smoke:
        if not args.config or not args.output_dir:
            parser.error('--smoke requires --config and --output-dir')
        trainer_smoke(args.config, args.output_dir, args.accelerator, args.devices, args.resume)
    else:
        unittest.main(argv=[__file__, *remaining])
