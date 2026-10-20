"""PhuongTU — normal-import architecture/loss regression checks for 6-LSH."""
import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import yaml

from models import get_backbone, get_criterion
from models.rpdnet.RepDWNet import RepDWNet

BASE_CONFIG = 'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml'
CONFIGS = ['configs/p3_scenarios/P3_LSH_L1_WDice.yaml',
           'configs/p3_scenarios/P3_LSH_L2_WDice.yaml']


def run(output):
    torch.set_num_threads(2)
    base_cfg = yaml.safe_load(Path(BASE_CONFIG).read_text())
    reports = []
    for level, config_path in enumerate(CONFIGS, 1):
        cfg = yaml.safe_load(Path(config_path).read_text())
        assert set(cfg) == set(base_cfg)
        for key in set(cfg) - {'experiment', 'backbone'}:
            assert cfg[key] == base_cfg[key], key
        assert {k: v for k, v in cfg['backbone'].items() if k != 'name'} == {
            k: v for k, v in base_cfg['backbone'].items() if k != 'name'}
        torch.manual_seed(base_cfg['seed'])
        baseline = get_backbone(base_cfg)
        torch.manual_seed(cfg['seed'])
        model = get_backbone(cfg)
        assert isinstance(model, RepDWNet)
        assert isinstance(model.down4[0], nn.MaxPool2d), 'LSH must retain the /16 stage'
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
        for module in (model.up1, model.up2, model.up3, model.up4):
            assert module.up.align_corners is True
        classifiers = [model.out_conv[0], model.head_s2]
        if level == 2:
            classifiers.append(model.head_s4)
        assert [c.in_channels for c in classifiers] == ([16, 16] if level == 1 else [16, 16, 32])
        assert all(c.out_channels == 3 and c.kernel_size == (1, 1) and c.bias is not None for c in classifiers)
        head_params = sum(p.numel() for c in classifiers for p in c.parameters())
        params = sum(p.numel() for p in model.parameters())
        assert head_params == (102 if level == 1 else 201)
        assert params == 189827 + (51 if level == 1 else 150)
        captured = {}
        def record(name):
            def hook(module, args, out):
                captured[name] = out
            return hook
        hooks = [getattr(model, name).register_forward_hook(record(name))
                 for name in ('up2', 'up3', 'up4', 'down4')]
        model.eval()
        baseline.eval()
        shapes = []
        for h, w in ((64, 64), (65, 79)):
            inputs = torch.randn(2, 3, h, w)
            with torch.no_grad():
                actual = model(inputs)
                f1, f2, f4 = captured['up4'], captured['up3'], captured['up2']
                expected = model.out_conv(f1) + F.interpolate(model.head_s2(f2), size=(h, w), mode='bilinear', align_corners=False)
                if level == 2:
                    expected = expected + F.interpolate(model.head_s4(f4), size=(h, w), mode='bilinear', align_corners=False)
                torch.testing.assert_close(actual, expected / (level + 1), rtol=0, atol=0)
                # Shared /1 readout remains exactly the L0 output with the same weights.
                torch.testing.assert_close(model.out_conv(f1), baseline(inputs), rtol=0, atol=0)
            assert isinstance(actual, torch.Tensor) and actual.shape == (2, 3, h, w)
            assert [f1.shape[1], f2.shape[1], f4.shape[1]] == [16, 16, 32]
            shapes.append(dict(input=[2, 3, h, w], output=list(actual.shape),
                               decoder_features=[list(f1.shape), list(f2.shape), list(f4.shape)],
                               deepest_shape=list(captured['down4'].shape)))
        for hook in hooks:
            hook.remove()
        model.train()
        target = torch.randint(0, 3, (2, 65, 79))
        target[:, :2, :3] = 255
        before = target.clone()
        logits = model(torch.randn(2, 3, 65, 79))
        criterion = get_criterion(cfg)
        loss = criterion(logits, target, mode='train', mask_keep=target != 255)
        assert torch.isfinite(loss)
        torch.testing.assert_close(target, before)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
        model.eval()
        deployed = copy.deepcopy(model)
        for module in list(deployed.modules()):
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        with torch.no_grad():
            probe = torch.randn(1, 3, 65, 79)
            reference, converted = model(probe), deployed(probe)
        error = float((reference - converted).abs().max())
        scale = max(float(reference.abs().max()), 1.)
        assert error / scale < 1e-5
        reports.append(dict(model=cfg['backbone']['name'], params=params, head_params=head_params,
                            shapes=shapes, finite_loss=float(loss.detach()), all_parameters_have_finite_gradients=True,
                            formula_and_baseline_features='passed',
                            deploy_max_abs_error=error, deploy_normalized_max_error=error / scale))
    import test_wdice
    original_tests = sorted(n for n in dir(test_wdice) if n.startswith('test_'))
    for name in original_tests:
        getattr(test_wdice, name)()
    # Verify chosen evaluation rule is backed by the real unchanged callbacks.
    import ast
    trainer_tree = ast.parse(Path('multi_metric_train.py').read_text())
    callbacks = [n for n in ast.walk(trainer_tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'ModelCheckpoint']
    assert any({k.arg: ast.literal_eval(k.value) for k in n.keywords if k.arg in ('monitor', 'mode')}
               == {'monitor': 'val_weed_iou', 'mode': 'max'} for n in callbacks)
    optimizer_tree = ast.parse(Path('models/model_multimetrics.py').read_text())
    configure = next(n for n in ast.walk(optimizer_tree)
                     if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'Adam' for n in ast.walk(configure))
    # Synthetic PNG fixture exercises the unchanged real loader/remap/crop.
    # This is not evidence of access to PhenoBench or GPU memory safety.
    import tempfile
    import numpy as np
    from PIL import Image
    from datasets import get_data_module
    with tempfile.TemporaryDirectory(prefix='lsh-data-') as fixture:
        data_root = Path(fixture)
        for split, shape in [('train', (768, 768)), ('val', (65, 79))]:
            for kind in ('images', 'semantics'):
                (data_root / split / kind).mkdir(parents=True)
            h, w = shape
            image = np.random.default_rng(17).integers(0, 256, (h, w, 3), dtype=np.uint8)
            raw = np.resize(np.array([0, 3, 4], dtype=np.uint8), (h, w))
            Image.fromarray(image).save(data_root / split / 'images/probe.png')
            Image.fromarray(raw).save(data_root / split / 'semantics/probe.png')
        (data_root / 'test/images').mkdir(parents=True)
        data_cfg = copy.deepcopy(base_cfg)
        data_cfg['data'].update(path_to_dataset=str(data_root), num_workers=0)
        data = get_data_module(data_cfg)
        data.setup('fit')
        train_sample, val_sample = data.train_ds[0], data.val_ds[0]
        assert list(train_sample['input_image'].shape) == [3, 768, 768]
        assert list(val_sample['input_image'].shape) == [3, 65, 79]
        expected = np.resize(np.array([0, 1, 2], dtype=np.int64), (65, 79))
        torch.testing.assert_close(val_sample['anno'], torch.from_numpy(expected))
        data_probe = dict(fixture='synthetic PNGs through unchanged PDC loader',
                          train_shape=list(train_sample['input_image'].shape),
                          val_shape=list(val_sample['input_image'].shape),
                          raw_label_remap_3_to_1_4_to_2='passed', actual_phenobench_tested=False)
    result = dict(author='PhuongTU', mode='normal-import', device='cpu', torch=torch.__version__,
                  reports=reports, original_wdice_tests=original_tests, data_probe=data_probe,
                  checkpoint_rule=dict(monitor='val_weed_iou', mode='max'), long_training_run=False)
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/lsh_core_checks.json')
    run(parser.parse_args().output)
