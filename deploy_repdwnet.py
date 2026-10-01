"""
Deploy & ONNX Export Pipeline for RepDWNet
Chuyển đổi 1 bước (1-Step Direct Deployment) từ Checkpoint huấn luyện
sang mô hình triển khai gọn nhẹ, đồng thời xuất file ONNX cho TensorRT (Jetson Orin Nano).

Usage:
    python deploy_repdwnet.py --config configs/b1_scenarios/B1_run1_repdwnet.yaml \
                              --weights checkpoints/best.pth \
                              --output_dir ./deploy_output \
                              --onnx
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from models import get_backbone
from models.rpdnet.RPD_Module import RPD_model_deploy


def deploy_and_export(config_path, weights_path=None, output_dir='./deploy_output', export_onnx=True):
    os.makedirs(output_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. Khởi tạo mô hình ở chế độ training cấu trúc
    cfg['backbone']['deploy'] = False
    model = get_backbone(cfg)

    # 2. Tải weights nếu có
    if weights_path is not None and os.path.exists(weights_path):
        print(f"[Deploy] Loading checkpoint from: {weights_path}")
        ckpt = torch.load(weights_path, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        # Loại bỏ prefix 'model.' nếu có từ PyTorch Lightning hoặc DataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('model.', '') if k.startswith('model.') else k
            clean_state_dict[new_k] = v
        model.load_state_dict(clean_state_dict, strict=True)
        print("  ✓ Checkpoint loaded successfully.")
    else:
        print("[Deploy] No checkpoint provided; running structural deploy on initialized model.")

    model.eval()

    # 3. Chuyển đổi sang Deploy Topology (1-Step Direct Deploy)
    print("[Deploy] Fusing multi-branch modules via switch_to_deploy()...")
    deploy_model = RPD_model_deploy(model, do_copy=True)
    deploy_model.eval()

    # 4. Kiểm tra tính tương đương số học trên dummy input
    dummy_input = torch.randn(1, 3, 768, 768)
    with torch.no_grad():
        out_train = model(dummy_input)
        out_deploy = deploy_model(dummy_input)
    max_diff = (out_train - out_deploy).abs().max().item()
    print(f"  ✓ Max numerical difference (pre-deploy vs post-deploy): {max_diff:.8e}")

    # 5. Lưu trọng số mô hình triển khai
    deploy_weights_path = os.path.join(output_dir, 'repdwnet_deploy.pth')
    torch.save(deploy_model.state_dict(), deploy_weights_path)
    print(f"  ✓ Saved deployed weights to: {deploy_weights_path}")

    # 6. Xuất mô hình ONNX nếu được yêu cầu
    if export_onnx:
        onnx_path = os.path.join(output_dir, 'repdwnet_deploy.onnx')
        print(f"[ONNX] Exporting to {onnx_path} (Opset 17, static shape 1x3x768x768)...")
        torch.onnx.export(
            deploy_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None  # Static shape tối ưu cho TensorRT trên Jetson
        )
        print(f"  ✓ Successfully exported ONNX model: {onnx_path}")

    print("\n[Deploy] All operations completed successfully.")
    return deploy_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deploy & Export RepDWNet")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('--weights', type=str, default=None, help="Path to checkpoint weights file")
    parser.add_argument('--output_dir', type=str, default='./deploy_output', help="Output directory")
    parser.add_argument('--onnx', action='store_true', default=True, help="Export to ONNX")
    args = parser.parse_args()

    deploy_and_export(args.config, args.weights, args.output_dir, args.onnx)
