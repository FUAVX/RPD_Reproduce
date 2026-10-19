"""AAA reference, integration and actual Lightning/DDP smoke checks.

python test_aaa.py
python test_aaa.py --smoke --config configs/p3_scenarios/P3_AAA_Ex1_Early.yaml --output-dir RUN/smoke --accelerator gpu --devices 2
python test_aaa.py --native-smoke --accelerator gpu --output-dir RUN/native
"""
import argparse
import copy
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml
from models import get_backbone, get_criterion
from models.rpdnet.aaa import AAAMaxBlurPool, tri3_filter
from models.rpdnet.downsampling import make_downsampling
from models.rpdnet.RPD_Module import RPD_model_deploy
from scripts.analyze_aaa import load_checkpoint

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml'
SCENARIOS = [(1, 'Early', [1, 2]), (2, 'Late', [3, 4]), (3, 'All', [1, 2, 3, 4])]


def reference_pool(channels):
    path = ROOT/'tests/reference/aaa/blurpool.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != '126e1379b39afa0ae88936810039b84e73ab3e1a7b59c9cbc7f340ee18843467':
        raise AssertionError('Official reference fixture differs from the audited commit.')
    spec = importlib.util.spec_from_file_location('aaa_official_reference', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return nn.Sequential(nn.MaxPool2d(2, 1, padding=0, ceil_mode=False),
                         module.BlurPool(channels, filt_size=3, pad_type='reflect', stride=2, pad_off=0))


class SyntheticSegmentation(Dataset):
    def __len__(self): return 8
    def __getitem__(self, index):
        g = torch.Generator().manual_seed(index)
        return {'input_image': torch.randn(3, 32, 32, generator=g),
                'anno': torch.randint(0, 3, (32, 32), generator=g)}


def trainer_smoke(config, output_dir, accelerator='cpu', devices=1, resume=False):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy
    from models.model_multimetrics import SegmentationNetwork
    from callbacks.config_callback import ConfigCallback
    from callbacks.training_cost import TrainingCostCallback
    cfg = yaml.safe_load(Path(config).read_text())
    pl.seed_everything(cfg['seed'])
    network = get_backbone(cfg)
    # Exercise actual initial buffer broadcast, rather than assuming it works.
    if int(os.environ.get('LOCAL_RANK', '0')) == 1 and not resume:
        next(m for m in network.modules() if isinstance(m, AAAMaxBlurPool)).filt.zero_()
    module = SegmentationNetwork(network, get_criterion(cfg), cfg['train']['learning_rate'], cfg['train']['weight_decay'],
        train_step_settings=['regular'], val_step_settings=['regular'],
        scheduler_type=cfg['train']['scheduler'], warmup_epochs=cfg['train']['warmup_epochs'])
    out = Path(output_dir);out.mkdir(parents=True, exist_ok=True)

    class StateProbe(Callback):
        def on_train_start(self, trainer, model):
            for pool in model.network.modules():
                if isinstance(pool, AAAMaxBlurPool):
                    assert torch.equal(pool.filt, tri3_filter(pool.channels, pool.filt.dtype, pool.filt.device))
        def on_train_end(self, trainer, model):
            import torch.distributed as dist
            digest = hashlib.sha256()
            for name, tensor in model.network.named_parameters():
                digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
            hashes = [digest.hexdigest()]
            if dist.is_initialized():
                hashes = [None]*trainer.world_size;dist.all_gather_object(hashes, digest.hexdigest())
            assert len(set(hashes)) == 1, 'DDP trainable weights differ across ranks'
            if trainer.is_global_zero:
                (out/'ddp_state_probe.json').write_text(json.dumps({'buffer_broadcast': 'passed', 'parameter_hashes': hashes}))

    ckpt = ModelCheckpoint(dirpath=str(out/'checkpoints'), save_last=True, monitor='val_weed_iou', mode='max')
    logger = TensorBoardLogger(str(out), name='lightning_logs', version=0, default_hp_metric=False)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=120)) if devices > 1 else 'auto',
        precision=32, max_epochs=2 if resume else 1,
        callbacks=[ConfigCallback(cfg), StateProbe(), TrainingCostCallback(out, cfg['experiment']['id']), ckpt],
        logger=logger, default_root_dir=str(out), check_val_every_n_epoch=1,
        limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
        enable_progress_bar=False, enable_model_summary=False, log_every_n_steps=1)
    loader = DataLoader(SyntheticSegmentation(), batch_size=2)
    trainer.fit(module, loader, loader, ckpt_path=str(out/'checkpoints/last.ckpt') if resume else None)
    if trainer.is_global_zero:
        raw = torch.load(out/'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
        assert raw['global_step'] == (4 if resume else 2)
        assert raw['optimizer_states'] and all(torch.isfinite(v).all() for v in raw['state_dict'].values())
        fresh = get_backbone(cfg);load_checkpoint(fresh, out/'checkpoints/last.ckpt')
        cost = json.loads((out/'training_cost.json').read_text())
        assert cost['world_size'] == devices and len(cost['ranks']) == devices
        if accelerator == 'gpu':
            assert all(r['peak_gpu_allocated_bytes'] > 0 for r in cost['ranks'])
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        assert EventAccumulator(logger.log_dir).Reload().Scalars('val_weed_iou')
        print(f'PASS AAA trainer smoke: {Path(config).stem}, {accelerator}, devices={devices}, resume={resume}')


def native_smoke(accelerator, output_dir):
    device = torch.device('cuda' if accelerator == 'gpu' else 'cpu')
    rows = []
    with torch.inference_mode():
        for ex, name, stages in SCENARIOS:
            cfg = yaml.safe_load((ROOT/f'configs/p3_scenarios/P3_AAA_Ex{ex}_{name}.yaml').read_text())
            torch.manual_seed(cfg['seed']);model = get_backbone(cfg).to(device).eval()
            for size in (768, 1024):
                x = torch.zeros(1, 3, size, size, device=device)
                y = model(x)
                assert y.shape == (1, 3, size, size) and torch.isfinite(y).all()
                rows.append({'placement': name, 'resolution': size, 'shape': list(y.shape), 'finite': True})
            del model, x, y
            if device.type == 'cuda':torch.cuda.empty_cache()
    path = Path(output_dir);path.mkdir(parents=True, exist_ok=True)
    (path/'native_shapes.json').write_text(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else None, 'results': rows}, indent=2))


class AAATests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA reference parity requires an actual GPU')
    def test_official_cuda_forward_and_backward(self):
        for channels, h, w in [(3,7,9),(16,768,768)]:
            m, ref = AAAMaxBlurPool(channels).cuda(), reference_pool(channels).cuda()
            x = torch.randn(1,channels,h,w,device='cuda',requires_grad=channels == 3)
            with torch.set_grad_enabled(channels == 3):
                y, expected = m(x), ref(x)
                torch.testing.assert_close(y,expected,rtol=0,atol=1e-6)
            if channels == 3:
                a=torch.autograd.grad(y.sum(),x,retain_graph=True)[0]
                b=torch.autograd.grad(expected.sum(),x)[0]
                torch.testing.assert_close(a,b,rtol=0,atol=1e-6)

    def test_official_forward_backward_and_geometry(self):
        for dtype in (torch.float32, torch.float64):
            for shape in [(8, 10), (7, 9), (3, 3)]:
                g = torch.Generator().manual_seed(1682409321)
                x = torch.randn(2, 3, *shape, generator=g, dtype=dtype, requires_grad=True)
                m, ref = AAAMaxBlurPool(3).to(dtype), reference_pool(3).to(dtype)
                y, expected = m(x), ref(x)
                self.assertEqual(y.shape[-2:], (shape[0]//2, shape[1]//2))
                torch.testing.assert_close(y, expected, atol=1e-6, rtol=0)
                a = torch.autograd.grad(y.sum(), x, retain_graph=True)[0]
                b = torch.autograd.grad(expected.sum(), x)[0]
                torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
        x = torch.randn(1, 1, 4, 5, dtype=torch.double, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(AAAMaxBlurPool(1).double(), (x,)))

    def test_buffers_constants_alignment_and_constraints(self):
        m = AAAMaxBlurPool(3)
        self.assertEqual(list(m.parameters()), [])
        self.assertEqual(list(m.state_dict()), ['filt'])
        self.assertFalse(m.filt.requires_grad)
        self.assertTrue(torch.equal(m.filt, tri3_filter(3)))
        for value in (-2., 3.):
            torch.testing.assert_close(m(torch.full((1, 3, 8, 8), value)), torch.full((1, 3, 4, 4), value))
        x = torch.arange(8.).repeat(8, 1)[None, None]
        torch.testing.assert_close(AAAMaxBlurPool(1)(x)[0, 0, 1], torch.tensor([1.5, 3., 5., 6.5]))
        with self.assertRaises(ValueError):m(torch.ones(1, 3, 2, 2))
        with self.assertRaises(ValueError):AAAMaxBlurPool(1, filt_size=4)
        with self.assertRaises(RuntimeError):m.load_state_dict({'filt': torch.zeros_like(m.filt)}, strict=True)

    def test_configs_rng_host_parameters_checkpoint_and_deploy(self):
        base = yaml.safe_load(BASE.read_text())
        torch.manual_seed(base['seed']);control = get_backbone(base).eval()
        expected_rng = torch.get_rng_state().clone()
        for ex, name, stages in SCENARIOS:
            cfg = yaml.safe_load((ROOT/f'configs/p3_scenarios/P3_AAA_Ex{ex}_{name}.yaml').read_text())
            inherited = copy.deepcopy(cfg);inherited['experiment'] = base['experiment'];del inherited['backbone']['downsampling']
            self.assertEqual(inherited, base)
            torch.manual_seed(base['seed']);m = get_backbone(cfg).eval()
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(sum(p.numel() for p in m.parameters()), 189827)
            for key, tensor in control.state_dict().items():self.assertTrue(torch.equal(tensor, m.state_dict()[key]), key)
            self.assertEqual([i for i in range(1,5) if isinstance(getattr(m,f'down{i}')[0], AAAMaxBlurPool)], stages)
            # Bounded initialized weights are an integration sanity case, not a
            # fabricated acceptance threshold for arbitrary trained checkpoints.
            with torch.no_grad():
                for layer in m.modules():
                    if isinstance(layer, nn.Conv2d):layer.weight.mul_(.1)
            x = torch.randn(1,3,33,35)
            fresh = get_backbone(cfg).eval();fresh.load_state_dict(m.state_dict(), strict=True)
            deploy = RPD_model_deploy(m, do_copy=True).eval()
            self.assertEqual(sum(isinstance(v, AAAMaxBlurPool) for v in deploy.modules()), len(stages))
            with torch.inference_mode():
                torch.testing.assert_close(fresh(x), m(x), atol=0, rtol=0)
                torch.testing.assert_close(deploy(x), m(x), atol=1e-4, rtol=1e-4)

    def test_dispatch_rejects_unapproved_config(self):
        good = dict(variant='aaa_tri3', stages=[1,2], filt_size=3, pad_type='reflect', stride=2, pad_off=0)
        for extra in [{'stages': [2,3]}, {'filt_size':4}, {'stride':1}, {'pad_off':True}, {'temperature':1}]:
            with self.assertRaises(ValueError):make_downsampling(16, {**good, **extra}, 1)

    def test_notebook_branch_modes_and_campaign_isolation(self):
        import ast
        nb = json.loads((ROOT/'notebook/kaggle_train_repdwnet_aaa.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(line for line in cell['source'] if not line.startswith('!')))
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        scope = {};exec(''.join(nb['cells'][1]['source']),scope)
        self.assertEqual(scope['GIT_REF'],'f/down-aaa')
        self.assertEqual(scope['DEVICES'],2)
        self.assertEqual(scope['EXPERIMENTS']['ex3'],('P3_AAA_Ex3_All',[1,2,3,4]))
        for scenario, stages in scope['EXPERIMENTS'].values():
            cfg = yaml.safe_load((ROOT/f'configs/p3_scenarios/{scenario}.yaml').read_text())
            self.assertEqual(cfg['backbone']['downsampling']['stages'],stages)
        launch = ''.join(nb['cells'][5]['source'])
        self.assertIn("'--devices', str(DEVICES)",launch)
        self.assertIn('scripts/benchmark_aaa.py',launch)
        self.assertIn('torch.distributed.run',launch)
        self.assertIn("'--resume'",launch)
        self.assertIn("'multi_metric_train.py'",launch)

    def test_actual_trainer_resume(self):
        cfg = ROOT/'configs/p3_scenarios/P3_AAA_Ex1_Early.yaml'
        with tempfile.TemporaryDirectory() as out:
            trainer_smoke(cfg, out);trainer_smoke(cfg, out, resume=True)
            self.assertEqual(len(list((Path(out)/'training_cost').glob('start*_rank0.json'))), 2)
            cost = json.loads((Path(out)/'training_cost.json').read_text())
            self.assertEqual(cost['start_epoch_0based'],1)
            self.assertEqual(cost['start_global_step'],2)
            self.assertEqual(cost['status'],'completed')


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--native-smoke', action='store_true')
    parser.add_argument('--config');parser.add_argument('--output-dir')
    parser.add_argument('--accelerator', choices=['cpu','gpu'], default='cpu')
    parser.add_argument('--devices', type=int, default=1);parser.add_argument('--resume', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.smoke:
        if not args.config or not args.output_dir:parser.error('Smoke requires config/output-dir')
        trainer_smoke(args.config, args.output_dir, args.accelerator, args.devices, args.resume)
    elif args.native_smoke:
        if not args.output_dir:parser.error('Native smoke requires output-dir')
        native_smoke(args.accelerator, args.output_dir)
    else:
        unittest.main(argv=[__file__, *remaining])
