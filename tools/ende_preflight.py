"""Run En/De shape/backward checks in a separate process before torchrun."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from models import get_backbone
from models.rpdnet.RepDWNet import RepDWNet, DoubleConv, DoubleConv_down_repdw, Down_repdw, Up_repdw
from models.rpdnet.RepDWNetEnDe import NotebookRepDWNetExp1, NotebookRepDWNetExp2
from utils.ende_training import check_dataset, validate_fresh_config


def check_model(cfg):
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    model = get_backbone(cfg).eval()
    expected_class = {'NotebookRepDWNetExp1': NotebookRepDWNetExp1,
                      'NotebookRepDWNetExp2': NotebookRepDWNetExp2}[cfg['backbone']['name']]
    assert type(model) is expected_class
    assert model.__class__.forward is RepDWNet.forward
    assert type(model.in_conv) is DoubleConv
    for i in range(1, 5):
        assert type(getattr(model, f'down{i}')) is Down_repdw
        up = getattr(model, f'up{i}')
        assert up.__class__.forward is Up_repdw.forward
        assert type(up.conv) is DoubleConv_down_repdw
        assert up.conv.mid_channels == cfg['backbone']['decoder_mid_channels'][i-1]
        assert up.conv.in_channels == cfg['backbone']['decoder_concat_channels'][i-1]
    baseline = RepDWNet(3, base_c=16, deploy=False, use_se=False, num_dw_branches=4)
    assert [(k, tuple(v.shape)) for k, v in model.in_conv.state_dict().items()] == [
        (k, tuple(v.shape)) for k, v in baseline.in_conv.state_dict().items()]
    if type(model) is NotebookRepDWNetExp2:
        for i in range(1, 5):
            assert [(k, tuple(v.shape)) for k, v in getattr(model, f'down{i}').state_dict().items()] == [
                (k, tuple(v.shape)) for k, v in getattr(baseline, f'down{i}').state_dict().items()]
    stage_names = ['in_conv'] + [f'down{i}' for i in range(1, 5)] + [f'up{i}' for i in range(1, 5)]
    shapes = {}
    handles = [getattr(model, name).register_forward_hook(
        lambda module, inputs, output, name=name: shapes.__setitem__(name, list(output.shape))) for name in stage_names]
    channels = [16] + cfg['backbone']['encoder_output_channels'] + cfg['backbone']['decoder_output_channels']
    results = []
    with torch.no_grad():
        for h, w in [(64, 64), (65, 67)]:
            output = model(torch.randn(1, 3, h, w))
            assert tuple(output.shape) == (1, 3, h, w)
            assert torch.isfinite(output).all()
            assert [shapes[name][1] for name in stage_names] == channels
            results.append(dict(input=[1, 3, h, w], output=list(output.shape), stages=dict(shapes)))
    for handle in handles:
        handle.remove()
    model.train()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 32, 32)),
                                             torch.zeros(2, 32, 32, dtype=torch.long))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Checkpoint must strict-load into the same experiment, then fuse existing B1 modules.
    restored = get_backbone(cfg).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 3, 33, 35)
    with torch.no_grad():
        before = restored(x)
        for module in list(restored.modules()):
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        after = restored(x)
    # Existing B1 RepDW fusion intentionally computes in FP32. Untrained
    # multi-branch logits have a large range; bound error against that range
    # rather than relative to individual logits close to zero.
    fusion_max_abs_error = float((before - after).abs().max())
    fusion_scaled_error = fusion_max_abs_error / max(1.0, float(before.abs().max()))
    assert fusion_scaled_error < 1e-5, fusion_scaled_error
    return dict(model=type(model).__name__, num_parameters=sum(p.numel() for p in model.parameters()),
                shape_checks=results, backward='passed', strict_load_and_fusion='passed',
                fusion_max_abs_error=fusion_max_abs_error, fusion_scaled_error=fusion_scaled_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--check-dataset', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    hashes = validate_fresh_config(cfg, {})
    result = check_model(cfg)
    result['block_sha256'] = hashes
    if args.check_dataset:
        result['dataset'] = check_dataset(cfg)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
