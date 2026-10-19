"""
Smoke test script for RepDWNet_DSP architecture variants.
Verifies:
  - Model initialization for all 4 variants (Ex1, Ex2, Ex3, Ex4).
  - Forward pass on dummy tensor with shapes (1, 3, 768, 768) and (1, 3, 1024, 1024).
  - Parameter counting.
  - Structural re-parameterization deploy check for Ex2.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from models import get_backbone
except ImportError:
    get_backbone = None

from models.rpdnet.detail_spatial_path import RepDWNet_DSP
from models.rpdnet.RPD_Module import RPD_model_deploy


def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST: RepDWNet_DSP Architecture Variants")
    print("=" * 70)

    variants = [
        ('dsp_ex1_lite', 'DSP-Ex1 (Canonical DW-Spatial)'),
        ('dsp_ex2_rep', 'DSP-Ex2 (Rep-Structural Multi-branch)'),
        ('dsp_ex3_ffm', 'DSP-Ex3 (BiSe-FFM Attention Gating)'),
        ('dsp_ex4_dual', 'DSP-Ex4 (PID-Bilateral Dual-Scale)'),
    ]

    for variant_key, variant_desc in variants:
        print(f"\n--- Testing: {variant_desc} ({variant_key}) ---")
        cfg = {
            'backbone': {
                'name': 'RepDWNet_DSP',
                'num_classes': 3,
                'pretrained': False,
                'num_dw_branches': 4,
                'use_se': False,
                'base_c': 16,
                'deploy': False,
                'dsp_variant': variant_key
            }
        }

        # 1. Instantiate model via get_backbone or direct class
        if get_backbone is not None:
            model = get_backbone(cfg)
        else:
            model = RepDWNet_DSP(
                num_classes=3,
                base_c=16,
                deploy=False,
                dsp_variant=variant_key
            )
        model.eval()

        # 2. Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Params: Total = {total_params:,} | Trainable = {trainable_params:,}")

        # 3. Test forward pass with smaller tensor first (768x768 - PhenoBench crop during train)
        x_train = torch.randn(1, 3, 768, 768)
        with torch.no_grad():
            out_train = model(x_train)
        print(f"  Forward pass (1, 3, 768, 768) -> Output: {tuple(out_train.shape)} - OK")
        assert out_train.shape == (1, 3, 768, 768), f"Expected (1, 3, 768, 768), got {out_train.shape}"

        # 4. Test forward pass with native resolution (1024x1024)
        x_native = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            out_native = model(x_native)
        print(f"  Forward pass (1, 3, 1024, 1024) -> Output: {tuple(out_native.shape)} - OK")
        assert out_native.shape == (1, 3, 1024, 1024), f"Expected (1, 3, 1024, 1024), got {out_native.shape}"

        # 5. For Ex2, test deploy re-parameterization
        if variant_key == 'dsp_ex2_rep':
            print("  Testing Structural Re-parameterization (deploy)...")
            # Verify DSP branch parity directly
            with torch.no_grad():
                f_dsp_pre = model.dsp(x_train)
            deploy_model = RPD_model_deploy(model, do_copy=True)
            deploy_model.eval()
            total_deploy_params = sum(p.numel() for p in deploy_model.parameters())
            print(f"  Deploy Params: {total_deploy_params:,} (Saved {total_params - total_deploy_params:,} params via collapse)")
            with torch.no_grad():
                f_dsp_post = deploy_model.dsp(x_train)
                out_deploy = deploy_model(x_train)
            dsp_diff = (f_dsp_pre - f_dsp_post).abs().max().item()
            net_diff = (out_train - out_deploy).abs().max().item()
            print(f"  DSP module max abs error: {dsp_diff:.6e}")
            print(f"  Full network max abs error (untrained init): {net_diff:.6e}")
            assert dsp_diff < 1e-4, f"DSP deploy numerical mismatch too high: {dsp_diff}"
            print("  ✓ Structural Re-parameterization parity verified successfully!")

        print(f"  ✓ {variant_key} PASSED all checks.")

    print("\n" + "=" * 70)
    print("ALL 4 REPDWNET_DSP VARIANTS PASSED SMOKE TEST SUCCESSFULLY!")
    print("=" * 70)


if __name__ == '__main__':
    run_smoke_test()
