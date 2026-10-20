"""Meaningful core regression checks. Use --core-only without training deps."""
import argparse
import ast
import copy
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch import nn


def run(core_only=False, output="p3_test_results.json"):
    if core_only:
        from tools.p3_core_loader import load_core
        RepDWNet, RepDWNetH8, RepDWNetH4H8Fusion, losses = load_core()
        # Execute the real factory function with its real classes, but do not
        # imply that eager imports / Lightning integration were exercised.
        factory_tree = ast.parse(Path("models/__init__.py").read_text(encoding="utf-8"))
        factory_function = next(n for n in factory_tree.body
                                if isinstance(n, ast.FunctionDef) and n.name == "get_backbone")
        namespace = dict(Dict=Dict, nn=nn, RepDWNet=RepDWNet,
                         RepDWNetH8=RepDWNetH8, RepDWNetH4H8Fusion=RepDWNetH4H8Fusion)
        exec(compile(ast.Module(body=[factory_function], type_ignores=[]),
                     "models/__init__.py", "exec"), namespace)
        get_backbone, get_criterion = namespace["get_backbone"], losses.get_criterion
    else:
        from models import get_backbone, get_criterion
        from models.rpdnet.RepDWNet import RepDWNet
        from models.rpdnet.RepDWNetH8 import RepDWNetH8
        from models.rpdnet.RepDWNetH4H8Fusion import RepDWNetH4H8Fusion

    torch.set_num_threads(2)
    torch.manual_seed(17)
    base_cfg = yaml.safe_load(Path("configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml").read_text())
    config_paths = ["configs/p3_scenarios/P3_SRR_H8_WDice.yaml",
                    "configs/p3_scenarios/P3_SRR_H4_H8_Fusion_WDice.yaml"]
    configs = [yaml.safe_load(Path(p).read_text()) for p in config_paths]
    for cfg in configs:
        assert set(cfg) == set(base_cfg), "Unexpected top-level recipe keys"
        for key in set(base_cfg) - {"experiment", "backbone"}:
            assert cfg[key] == base_cfg[key], f"Unexpected recipe change: {key}"

    baseline = get_backbone(base_cfg)
    h8 = get_backbone(configs[0])
    fusion = get_backbone(configs[1])
    assert isinstance(h8.down4[0], nn.Identity) and isinstance(h8.up1.up, nn.Identity)
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in h8.parameters())
    assert list(baseline.state_dict()) == list(h8.state_dict())
    assert len(fusion.exchanges) == 2
    final_cfg = copy.deepcopy(configs[1])
    final_cfg["backbone"]["exchange_mode"] = "final_only"
    final_only = get_backbone(final_cfg)
    assert list(final_only.exchanges.keys()) == ["1"]
    exchange = fusion.exchanges["0"].eval()
    high, low = torch.randn(2, 32, 17, 19), torch.randn(2, 128, 8, 9)
    with torch.no_grad():
        from_low = torch.nn.functional.interpolate(exchange.low_to_high(low), size=high.shape[-2:],
                                                   mode="bilinear", align_corners=True)
        from_high = torch.nn.functional.interpolate(exchange.high_to_low(high), size=low.shape[-2:],
                                                    mode="bilinear", align_corners=True)
        high_new, low_new = exchange(high, low)
        torch.testing.assert_close(high_new, torch.relu(high + from_low))
        torch.testing.assert_close(low_new, torch.relu(low + from_high))
    exchange.train()
    reports = []
    for name, model, cfg in (("baseline", baseline, base_cfg), ("h8", h8, configs[0]),
                             ("h4_h8", fusion, configs[1]), ("final_only", final_only, final_cfg)):
        shapes = {}
        hooks = []
        for stage in ("down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"):
            hooks.append(getattr(model, stage).register_forward_hook(
                lambda m, a, out, key=stage: shapes.__setitem__(key, list(out.shape))))
        model.train()
        logits = model(torch.randn(2, 3, 64, 64))
        target = torch.randint(0, 3, (2, 64, 64))
        target[:, 0, 0] = 255
        criterion = get_criterion(cfg)
        loss = criterion(logits, target, mode="train", mask_keep=target != 255)
        assert logits.shape == (2, 3, 64, 64) and torch.isfinite(loss)
        loss.backward()
        unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        invalid = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not unused, f"Unused trainable parameters in {name}: {unused}"
        assert not invalid, f"Non-finite gradients in {name}: {invalid}"
        assert shapes["down3"][-2:] == [8, 8]
        if name in ("baseline", "h8"):
            assert shapes["down4"][-2:] == ([4, 4] if name == "baseline" else [8, 8])
        assert shapes["up1"][-2:] == [8, 8]
        for hook in hooks:
            hook.remove()
        model.eval()
        with torch.no_grad():
            odd_shape = list(model(torch.randn(1, 3, 65, 79)).shape)
        assert odd_shape == [1, 3, 65, 79]
        # Verify structural reparameterization of the supplied blocks still
        # preserves the full variant's output, including unfused exchange convs.
        deployed = copy.deepcopy(model)
        for module in list(deployed.modules()):
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        probe = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            before, after = model(probe), deployed(probe)
        deploy_error = float((before-after).abs().max())
        output_scale = max(float(before.abs().max()), 1.0)
        print(f"Deploy diagnostic {name}: max_abs={deploy_error:.8g}, output_scale={output_scale:.8g}", flush=True)
        # Uncalibrated multi-branch random networks can have logits > 1e4.
        # Report absolute error too; this is a scale-normalized FP32 diagnostic,
        # not certification of per-pixel tolerance on a trained checkpoint.
        assert deploy_error / output_scale < 1e-5, f"Large normalized deploy error: {name}"
        reports.append(dict(name=name, params=sum(p.numel() for p in model.parameters()),
                            train_loss=float(loss.detach()), stage_shapes=shapes,
                            odd_output=odd_shape, all_trainable_parameters_used=True,
                            deploy_max_abs_error=deploy_error,
                            deploy_output_scale=output_scale,
                            deploy_normalized_max_error=deploy_error / output_scale))
        print(f"PASS: {name}", flush=True)
    # Re-run the original loss tests unchanged against the actual loss file.
    import test_wdice
    for test_name in sorted(n for n in dir(test_wdice) if n.startswith("test_")):
        getattr(test_wdice, test_name)()
    result = dict(mode="core-only" if core_only else "normal-import", torch=torch.__version__,
                  device="cpu", input=[2, 3, 64, 64], reports=reports,
                  original_wdice_checks="passed", full_training_run=False)
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"All Phase 3 checks passed; results: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-only", action="store_true")
    parser.add_argument("--output", default="p3_test_results.json")
    args = parser.parse_args()
    run(args.core_only, args.output)
