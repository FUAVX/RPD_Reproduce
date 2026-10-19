"""Run `python test_dpp.py` before launching a P3 Kaggle run."""

import copy
import json
import tempfile
from pathlib import Path

import torch
import yaml

from models import get_backbone
from models.rpdnet.dpp import LiteDPPSym, make_downsampling, dpp_optimizer_groups
from models.rpdnet.RPD_Module import RPD_model_deploy


def test_author_uniform_initialization():
    for channels in (16, 32, 64, 128):
        bound = channels ** -0.5
        torch.manual_seed(1682409321)
        expected_alpha = torch.empty(channels).uniform_(-bound, bound)
        expected_lambda = torch.empty(channels).uniform_(-bound, bound)
        torch.manual_seed(1682409321)
        layer = LiteDPPSym(channels)
        torch.testing.assert_close(layer.log_alpha, expected_alpha, rtol=0, atol=0)
        torch.testing.assert_close(layer.log_lambda, expected_lambda, rtol=0, atol=0)
        for parameter in (layer.log_alpha, layer.log_lambda):
            assert torch.all((parameter >= -bound) & (parameter <= bound))
            assert parameter.unique().numel() > 1
        assert not torch.equal(layer.log_alpha, layer.log_lambda)
        torch.testing.assert_close(layer.alpha, expected_alpha.exp(), rtol=0, atol=0)
        torch.testing.assert_close(layer.exponent, expected_lambda.exp(), rtol=0, atol=0)


def test_formula_shapes_and_limits():
    layer = LiteDPPSym(2).double()
    with torch.no_grad():
        layer.log_alpha.copy_(torch.tensor([0.2, -0.7]))
        layer.log_lambda.copy_(torch.tensor([0.5, 0.9]))
    x = torch.randn(2, 2, 7, 9, dtype=torch.double)
    actual = layer(x)
    expected = torch.empty(2, 2, 3, 4, dtype=torch.double)
    for b in range(2):
        for c in range(2):
            for h in range(3):
                for w in range(4):
                    window = x[b, c, 2*h:2*h+2, 2*w:2*w+2]
                    energy = (window - window.mean()).square().add(0.001)
                    reward = (energy / energy.mean()).pow(layer.exponent[c] / 2)
                    weights = layer.alpha[c] + reward
                    expected[b, c, h, w] = (weights * window).sum() / weights.sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer(torch.full((1, 2, 4, 6), 3.0, dtype=torch.double)),
                               torch.full((1, 2, 2, 3), 3.0, dtype=torch.double))
    # lambda -> 0 yields an average, including negative activations.
    with torch.no_grad():
        layer.log_lambda.fill_(-30)
    torch.testing.assert_close(layer(x), torch.nn.functional.avg_pool2d(x, 2))
    # Large finite exponents should not overflow reward normalization.
    with torch.no_grad():
        layer.log_lambda.fill_(8)
    assert torch.isfinite(layer(x)).all()


def test_author_graph_forward_and_gradients():
    # Independent reference follows the released avgpool/nearest-upsample graph.
    torch.manual_seed(23)
    layer = LiteDPPSym(3).double()
    x = torch.randn(2, 3, 6, 8, dtype=torch.double, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_alpha = layer.log_alpha.detach().clone().requires_grad_()
    reference_lambda = layer.log_lambda.detach().clone().requires_grad_()
    pool = torch.nn.functional.avg_pool2d
    up = lambda value: torch.nn.functional.interpolate(value, scale_factor=2, mode='nearest')
    difference = reference_x - up(pool(reference_x, 2))
    energy = difference.square() + 0.001
    normalized = energy / up(pool(energy, 2))
    weights = normalized.pow(reference_lambda.exp().view(1, 3, 1, 1) / 2)
    weights = weights + reference_alpha.exp().view(1, 3, 1, 1)
    expected = pool(reference_x * weights, 2) / pool(weights, 2)
    actual = layer(x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, (x, layer.log_alpha, layer.log_lambda), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_x, reference_alpha, reference_lambda), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)


def test_author_alpha_projection():
    layer = LiteDPPSym(1).double()
    with torch.no_grad():
        layer.log_alpha.fill_(12)
        layer.log_lambda.zero_()
    # Logging must reflect capped effective alpha, even before the next forward.
    torch.testing.assert_close(layer.alpha, torch.tensor([10.], dtype=torch.double).exp())
    x = torch.tensor([[[[0., 1.], [2., 8.]]]], dtype=torch.double)
    layer(x).sum().backward()
    assert layer.log_alpha.item() == 10
    assert layer.log_alpha.grad is not None and layer.log_alpha.grad.abs().sum() > 0
    # Repeated forwards in the normal range must support a combined backward.
    layer.zero_grad()
    (layer(x).sum() + layer(x * 2).sum()).backward()
    assert torch.isfinite(layer.log_alpha.grad).all()


def test_gradients_and_optimizer():
    torch.manual_seed(17)
    layer = LiteDPPSym(2).double()
    x = torch.randn(1, 2, 4, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))
    model = torch.nn.Sequential(layer, torch.nn.Conv2d(2, 3, 1).double())
    groups = dpp_optimizer_groups(model, 2e-4, 2e-4)
    all_ids = [id(p) for group in groups for p in group['params']]
    assert len(all_ids) == len(set(all_ids)) == len(list(model.parameters()))
    assert {id(p) for p in groups[1]['params']} == {id(layer.log_alpha), id(layer.log_lambda)}
    assert groups[0]['weight_decay'] == 2e-4 and groups[1]['weight_decay'] == 0
    before = [p.detach().clone() for p in layer.parameters()]
    optimizer = torch.optim.Adam(groups)
    for _ in range(3):
        optimizer.zero_grad()
        model(x).square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in layer.parameters())
        optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, layer.parameters()))


def test_configs_placement_and_deploy():
    root = Path(__file__).resolve().parent
    baseline = yaml.safe_load((root / 'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
    torch.manual_seed(1)
    control = get_backbone(baseline).eval()
    explicit = copy.deepcopy(baseline)
    explicit['backbone']['downsampling'] = {'variant': 'maxpool', 'stages': []}
    torch.manual_seed(1)
    unchanged = get_backbone(explicit).eval()
    assert control.state_dict().keys() == unchanged.state_dict().keys()
    x = torch.randn(1, 3, 33, 35)
    with torch.no_grad():
        torch.testing.assert_close(control(x), unchanged(x), rtol=0, atol=0)
    for number, name, stages, extra in [(1, 'Early', [1, 2], 96), (2, 'Late', [3, 4], 384),
                                        (3, 'All', [1, 2, 3, 4], 480)]:
        config = yaml.safe_load((root / f'configs/p3_scenarios/P3_DPP_Ex{number}_{name}.yaml').read_text())
        inherited = copy.deepcopy(config)
        del inherited['backbone']['downsampling']
        inherited['experiment'] = baseline['experiment']
        assert inherited == baseline, 'P3 changed settings outside downsampling/experiment metadata.'
        model = get_backbone(config).eval()
        # Keep the untrained multi-branch network in a bounded activation range
        # and exercise fusion with non-default BN statistics (no trained CKPT needed).
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.weight.mul_(0.1)
                elif isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.uniform_(-0.1, 0.1)
                    module.running_var.uniform_(0.7, 1.3)
        actual_stages = [i for i in range(1, 5) if isinstance(getattr(model, f'down{i}')[0], LiteDPPSym)]
        assert actual_stages == stages
        assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in control.parameters()) == extra
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 33, 35)
        restored = get_backbone(config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        deployed = RPD_model_deploy(model, do_copy=True).eval()
        deploy_config = copy.deepcopy(config)
        deploy_config['backbone']['deploy'] = True
        reloaded_deploy = get_backbone(deploy_config).eval()
        reloaded_deploy.load_state_dict(deployed.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(x), y, rtol=0, atol=0)
            torch.testing.assert_close(deployed(x), y, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(reloaded_deploy(x), deployed(x), rtol=0, atol=0)
        assert sum(isinstance(m, LiteDPPSym) for m in deployed.modules()) == len(stages)
    for config in ({'variant': 'lite_dpp_asym', 'stages': [1]},
                   {'variant': 'full_dpp', 'stages': [1]},
                   {'variant': 'lite_dpp_sym', 'stages': [5]},
                   {'variant': 'maxpool', 'stages': [1]}):
        try:
            make_downsampling(16, config, 1)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid/unsupported configuration accepted: {config}')


def test_logging_training_and_checkpoint_exports():
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset
    from callbacks.dpp_logging import DPPLoggingCallback

    class TinyRun(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.network = torch.nn.Sequential(LiteDPPSym(2), torch.nn.Conv2d(2, 1, 1))

        def training_step(self, batch, batch_idx):
            return self.network(batch[0]).square().mean()

        def validation_step(self, batch, batch_idx):
            self.log('val_weed_iou', self.network(batch[0]).mean())

        def configure_optimizers(self):
            return torch.optim.Adam(dpp_optimizer_groups(self, 2e-4, 2e-4))

    with tempfile.TemporaryDirectory() as directory:
        callback = DPPLoggingCallback(directory, 'test')
        checkpoint = ModelCheckpoint(dirpath=directory, monitor='val_weed_iou', mode='max', save_last=True)
        trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=2, logger=False,
                             callbacks=[checkpoint, callback], enable_progress_bar=False,
                             enable_model_summary=False, num_sanity_val_steps=0)
        loader = DataLoader(TensorDataset(torch.randn(4, 2, 4, 4)), batch_size=2)
        trainer.fit(TinyRun(), loader, loader)
        out = Path(directory) / 'dpp_parameters'
        history = [json.loads(line) for line in (out / 'history.jsonl').read_text().splitlines()]
        assert [record['event'] for record in history] == ['initial', 'epoch_end', 'epoch_end', 'final']
        assert len(history[-1]['layers']['0']['alpha']['values']) == 2
        for label, path in [('val_weed_iou', checkpoint.best_model_path), ('last', checkpoint.last_model_path)]:
            exported = json.loads((out / f'checkpoint_{label}.json').read_text())
            state = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
            torch.testing.assert_close(torch.tensor(exported['layers']['0']['alpha']['values']),
                                       state['network.0.log_alpha'].exp())
        assert len(json.loads((out / 'training_cost.json').read_text())['epoch_wall_seconds_including_validation']) == 2


if __name__ == '__main__':
    torch.set_num_threads(2)
    for test in (test_author_uniform_initialization, test_formula_shapes_and_limits,
                 test_author_graph_forward_and_gradients, test_author_alpha_projection, test_gradients_and_optimizer,
                 test_configs_placement_and_deploy, test_logging_training_and_checkpoint_exports):
        test()
        print(f'PASS: {test.__name__}')
    print('DPP checks passed.')
