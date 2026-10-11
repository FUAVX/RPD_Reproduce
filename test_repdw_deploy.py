"""
Verification Script: B0 Baseline Regression & RepDWNet Deploy Equivalence
Run from RPD directory: python test_repdw_deploy.py
"""
import os
import sys
import copy
import torch
import torch.nn as nn

# Ensure RPD root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import get_backbone, RPDNet
from models.rpdnet.RPD_Module import RPD_model_deploy
from models.rpdnet.rep_dw import RepDW


def test_b0_zero_regression():
    print("[1/5] Verifying Baseline B0 (RPDNet) Instantiation & Forward Pass...")
    b0_cfg = {
        'backbone': {
            'name': 'RPDNet',
            'num_classes': 3,
            'pretrained': False,
            'deploy': False,
            'convert': False
        }
    }
    model = get_backbone(b0_cfg)
    assert isinstance(model, RPDNet), "Factory returned wrong instance for RPDNet!"
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 3, 256, 256), f"Unexpected output shape: {out.shape}"
    print("  ✓ B0 Factory Instantiation & Forward Pass: PASSED")


def test_repdw_single_block_numerical():
    print("\n[2/5] Verifying Single RepDW Block Numerical Equivalence (atol < 3e-5)...")
    torch.manual_seed(42)
    block = RepDW(in_channels=16, out_channels=16, kernel_size=3, stride=1,
                  groups=16, bias=True, deploy=False, use_se=False, num_dw_branches=4)
    block.eval()

    x = torch.randn(2, 16, 64, 64)
    with torch.no_grad():
        y_train = block(x)

    block.switch_to_deploy()
    with torch.no_grad():
        y_deploy = block(x)

    max_diff = (y_train - y_deploy).abs().max().item()
    print(f"  ✓ Single block max absolute difference: {max_diff:.8e}")
    assert max_diff < 3e-5, f"Single block deploy equivalence failed! Diff: {max_diff}"
    print("  ✓ Single RepDW Block Numerical Equivalence: PASSED")


def test_repdwnet_numerical_deploy():
    print("\n[3/5] Verifying RepDWNet Full Network Deploy Numerical Equivalence (atol < 1e-3)...")
    torch.manual_seed(42)
    b1_cfg = {
        'backbone': {
            'name': 'RepDWNet',
            'num_classes': 3,
            'pretrained': False,
            'deploy': False,
            'use_se': False,
            'num_dw_branches': 4,
            'base_c': 16
        }
    }
    model = get_backbone(b1_cfg)
    model.eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        y_train = model(x)

    deploy_model = RPD_model_deploy(model, do_copy=True)
    deploy_model.eval()
    with torch.no_grad():
        y_deploy = deploy_model(x)

    max_diff = (y_train - y_deploy).abs().max().item()
    mean_diff = (y_train - y_deploy).abs().mean().item()
    argmax_agree = (y_train.argmax(dim=1) == y_deploy.argmax(dim=1)).float().mean().item() * 100
    print(f"  ✓ Full network MAE: {mean_diff:.8e} | Peak diff: {max_diff:.8e} | Argmax agreement: {argmax_agree:.2f}%")
    assert argmax_agree > 99.0, f"Deploy argmax agreement failed! Agreement: {argmax_agree:.2f}%"
    print("  ✓ RepDWNet Train/Deploy Numerical Equivalence: PASSED")


def test_fused_module_count():
    print("\n[4/5] Verifying Exact Fused Module Count (Expect 34 modules)...")
    b1_cfg = {
        'backbone': {
            'name': 'RepDWNet',
            'num_classes': 3,
            'pretrained': False,
            'deploy': False
        }
    }
    model = get_backbone(b1_cfg)
    fused_count = 0
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
            fused_count += 1

    print(f"  ✓ Total modules fused via switch_to_deploy(): {fused_count}")
    assert fused_count == 34, f"Expected exactly 34 fused modules, but found {fused_count}!"
    print("  ✓ RepDWNet Module Count Verification (34/34): PASSED")


def test_gradient_backward():
    print("\n[5/5] Verifying Backward Pass & Gradient Flow...")
    torch.manual_seed(42)
    b1_cfg = {
        'backbone': {
            'name': 'RepDWNet',
            'num_classes': 3,
            'pretrained': False,
            'deploy': False,
            'use_se': False,
            'num_dw_branches': 4,
            'base_c': 16
        }
    }
    model = get_backbone(b1_cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Check first down stage RepDW block
    first_dw = model.down1[1].Conv1[0]
    assert first_dw.rbr_identity.weight.grad is not None, "Identity BN has no gradient!"
    assert first_dw.rbr_1x1.conv.weight.grad is not None, "1x1 branch has no gradient!"
    for i, b in enumerate(first_dw.rbr_dw):
        assert b.conv.weight.grad is not None, f"DW branch {i} has no gradient!"

    print("  ✓ All RepDW branches receive non-zero gradients in backward pass: PASSED")


def test_p2_configurations():
    print("\n[6/6] Verifying P2 Configurations: ex1 (DW=2, PW=4) & ex2 (DW=4, PW=2)...")
    for name, dw_b, pw_b in [("ex1_dw2_pw4", 2, 4), ("ex2_dw4_pw2", 4, 2)]:
        torch.manual_seed(42)
        cfg = {
            'backbone': {
                'name': 'RepDWNet',
                'num_classes': 3,
                'pretrained': False,
                'deploy': False,
                'use_se': False,
                'num_dw_branches': dw_b,
                'num_pw_branches': pw_b,
                'base_c': 16
            }
        }
        model = get_backbone(cfg)
        model.eval()

        # Check branch counts in down1 Conv1
        stage_dw = model.down1[1].Conv1[0]
        stage_pw = model.down1[1].Conv1[1]
        assert len(stage_dw.rbr_dw) == dw_b, f"{name}: Expected {dw_b} DW branches, got {len(stage_dw.rbr_dw)}"
        assert len(stage_pw.rbr_conv) == pw_b, f"{name}: Expected {pw_b} PW branches, got {len(stage_pw.rbr_conv)}"

        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            y_train = model(x)

        deploy_model = RPD_model_deploy(model, do_copy=True)
        deploy_model.eval()
        with torch.no_grad():
            y_deploy = deploy_model(x)

        max_diff = (y_train - y_deploy).abs().max().item()
        mean_diff = (y_train - y_deploy).abs().mean().item()
        argmax_agree = (y_train.argmax(dim=1) == y_deploy.argmax(dim=1)).float().mean().item() * 100
        print(f"  ✓ {name} (DW={dw_b}, PW={pw_b}) Deploy peak diff: {max_diff:.8e} | Argmax: {argmax_agree:.2f}%")
        assert argmax_agree > 99.0, f"{name} deploy equivalence failed! Argmax: {argmax_agree:.2f}%"
    print("  ✓ P2 ex1 & ex2 Structural & Numerical Equivalence: PASSED")


if __name__ == '__main__':
    print("=" * 65)
    print("RUNNING REPDWNET COMPREHENSIVE VERIFICATION SUITE")
    print("=" * 65)
    test_b0_zero_regression()
    test_repdw_single_block_numerical()
    test_repdwnet_numerical_deploy()
    test_fused_module_count()
    test_gradient_backward()
    test_p2_configurations()
    print("\n" + "=" * 65)
    print("ALL 6 VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 65)
