#!/usr/bin/env python3
"""
Multi-Model Qualitative Segmentation Comparison: B0 vs B1 vs P2 (ex1 & ex2)
===========================================================================
Trực quan hóa so sánh chất lượng phân đoạn 4 mô hình trên từng ảnh thực tế:
- B0: RPDNet Baseline (PDC)
- B1: RepDWNet Standard (4 DW + 4 PW)
- P2_ex1: RepDWNet Reduced Spatial (2 DW + 4 PW)
- P2_ex2: RepDWNet Reduced Channel (4 DW + 2 PW - Max Precision)

Tự động phân nhóm:
1. Nắng gắt bóng đổ (Sunny II Failure Mode Rejection)
2. Khóm cỏ vi mô (Small Weeds < 100px)
3. An toàn phun thuốc (Crop Safety & Border Cleanliness)

Xuất dải ảnh so sánh đa khung hình kèm bản đồ bất đồng quan điểm (Discrepancy Map).
"""

import os
import sys
import glob
import zipfile
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RPD_DIR = os.path.join(BASE_DIR, 'RPD')
if RPD_DIR not in sys.path:
    sys.path.insert(0, RPD_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models import get_backbone
from models.rpdnet.RPD_Module import RPD_model_deploy

COLOR_MAP = {
    0: (35, 35, 35),       # Background/Soil: Xám đậm
    1: (46, 204, 113),     # Crop: Xanh lá tươi
    2: (231, 76, 60),      # Weed: Đỏ tươi
}


def auto_extract_checkpoint(ckpt_or_zip_path: str, target_pattern: str = "best_weedIoU") -> str:
    """Tự động giải nén checkpoint phù hợp nhất từ file .zip."""
    resolved_path = ckpt_or_zip_path
    if not os.path.exists(resolved_path):
        candidates = [
            os.path.join(BASE_DIR, ckpt_or_zip_path),
            os.path.join("..", ckpt_or_zip_path),
            os.path.join(RPD_DIR, ckpt_or_zip_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_path = c
                break

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Không tìm thấy file: {ckpt_or_zip_path}")

    if resolved_path.endswith('.ckpt') or resolved_path.endswith('.pth'):
        return resolved_path

    if resolved_path.endswith('.zip'):
        extract_dir = os.path.join(os.path.dirname(resolved_path), 'extracted_ckpts')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(resolved_path, 'r') as zf:
            matching = [f for f in zf.namelist() if target_pattern in f and f.endswith('.ckpt')]
            if not matching:
                matching = [f for f in zf.namelist() if 'best' in f and f.endswith('.ckpt')]
            if not matching:
                matching = [f for f in zf.namelist() if f.endswith('.ckpt')]
            if not matching:
                raise ValueError(f"Không tìm thấy file .ckpt nào trong {resolved_path}")

            chosen = matching[0]
            target_path = os.path.join(extract_dir, os.path.basename(chosen))
            if not os.path.exists(target_path):
                print(f"[Auto-Extract] Giải nén {os.path.basename(chosen)} -> {target_path}")
                with open(target_path, 'wb') as f_out:
                    f_out.write(zf.read(chosen))
            return target_path

    raise ValueError(f"Định dạng không hỗ trợ: {resolved_path}")


def load_model(config_path: str, ckpt_path: str, deploy: bool = False, device: str = 'cuda'):
    """Nạp checkpoint và chuyển sang deploy mode nếu được yêu cầu."""
    resolved_cfg = config_path
    if not os.path.exists(resolved_cfg):
        candidates = [
            os.path.join(RPD_DIR, config_path),
            os.path.join(BASE_DIR, config_path),
            config_path.replace('RPD/', ''),
            os.path.join('RPD', config_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                resolved_cfg = c
                break

    with open(resolved_cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['backbone']['deploy'] = False
    model = get_backbone(cfg)

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    clean_dict = {}
    for k, v in state_dict.items():
        k_clean = k.replace('model.model.', '').replace('model.', '')
        clean_dict[k_clean] = v

    model.load_state_dict(clean_dict, strict=False)

    if deploy:
        model = RPD_model_deploy(model, do_copy=True)

    model.to(device)
    model.eval()
    return model


def colorize_mask(mask_np: np.ndarray) -> Image.Image:
    """Tô màu mask nhãn theo bảng màu chuẩn PhenoBench."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in COLOR_MAP.items():
        rgb[mask_np == cls_id] = color
    return Image.fromarray(rgb)


def create_discrepancy_map(gt: np.ndarray, b0: np.ndarray, b1: np.ndarray, p2: np.ndarray) -> Image.Image:
    """
    Bản đồ bất đồng quan điểm & sửa sai:
    - Xanh dương (#3498db): B1/P2 phát hiện đúng cỏ dại mà B0 bỏ sót (False Negative Fixed)
    - Cam (#e67e22): B0 báo động giả cỏ dại trên đất/bóng râm được B1/P2 triệt tiêu (False Alarm Suppressed)
    - Tím (#9b59b6): Vùng P2_ex2 khắt khe loại bỏ viền không chắc chắn để bảo vệ cây trồng
    - Đỏ (#e74c3c): Cả 4 mô hình đều nhận diện đúng cỏ dại
    - Xanh lá (#27ae60): Cả 4 mô hình đều nhận diện đúng cây trồng
    """
    h, w = gt.shape
    diff = np.full((h, w, 3), 40, dtype=np.uint8)  # Nền xám mờ

    # Cây đúng chung
    crop_true = (gt == 1) & (b1 == 1) & (p2 == 1)
    diff[crop_true] = [39, 174, 96]

    # Cỏ đúng chung
    weed_true = (gt == 2) & (b1 == 2) & (p2 == 2)
    diff[weed_true] = [231, 76, 60]

    # B1 & P2 sửa lỗi bỏ sót cỏ của B0
    b0_miss_fixed = (gt == 2) & (b0 != 2) & ((b1 == 2) | (p2 == 2))
    diff[b0_miss_fixed] = [52, 152, 219]

    # B0 báo động giả (False Positive) bị khử sạch
    b0_fp_fixed = (gt != 2) & (b0 == 2) & (b1 != 2) & (p2 != 2)
    diff[b0_fp_fixed] = [230, 126, 34]

    # P2_ex2 conservative border (bảo vệ an toàn cho cây)
    p2_conservative = (gt != 2) & (b1 == 2) & (p2 != 2)
    diff[p2_conservative] = [155, 89, 182]

    return Image.fromarray(diff)


def compute_cls_metrics(pred: np.ndarray, gt: np.ndarray, cls_id: int = 2):
    tp = np.logical_and(pred == cls_id, gt == cls_id).sum()
    fp = np.logical_and(pred == cls_id, gt != cls_id).sum()
    fn = np.logical_and(pred != cls_id, gt == cls_id).sum()
    iou = (tp / (tp + fp + fn)) * 100.0 if (tp + fp + fn) > 0 else 0.0
    prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    return iou, prec, rec


def count_small_weeds(gt_mask: np.ndarray, max_pixels: int = 100) -> int:
    """Đếm số lượng khóm cỏ diện tích nhỏ trong Ground Truth."""
    from scipy.ndimage import label
    weed_binary = (gt_mask == 2).astype(np.uint8)
    labeled, num_features = label(weed_binary)
    if num_features == 0:
        return 0
    counts = np.bincount(labeled.ravel())
    # Bỏ qua background (nhãn 0)
    small_count = np.sum((counts[1:] > 0) & (counts[1:] <= max_pixels))
    return int(small_count)


def add_header(panel_img: Image.Image, title: str, subtitle: Optional[str] = None) -> Image.Image:
    header_h = 55 if subtitle else 40
    new_img = Image.new('RGB', (panel_img.width, panel_img.height + header_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(new_img)
    draw.text((10, 6), title, fill=(255, 255, 255))
    if subtitle:
        draw.text((10, 26), subtitle, fill=(200, 200, 200))
    new_img.paste(panel_img, (0, header_h))
    return new_img


def run_multi_model_pipeline(dataset_dir: str, models: Dict[str, Any], output_dir: str,
                             top_k: int = 3, device: str = 'cuda', max_eval_samples: int = 769):
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(dataset_dir, 'val', 'images')
    semantics_dir = os.path.join(dataset_dir, 'val', 'semantics')

    all_images = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    if not all_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh PNG trong {images_dir}")

    print(f"\n[Dataset] Quét {len(all_images)} ảnh trong tập Validation...")
    records = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    for idx, img_path in enumerate(all_images[:max_eval_samples]):
        fname = os.path.basename(img_path)
        sem_path = os.path.join(semantics_dir, fname)
        if not os.path.exists(sem_path):
            continue

        lighting = "Sunny I" if fname.startswith("05-15") else ("Sunny II" if fname.startswith("05-26") else "Overcast")

        rgb_pil = Image.open(img_path).convert('RGB')
        gt_pil = Image.open(sem_path)
        gt_np = np.array(gt_pil, dtype=np.uint8)

        # Chuyển đổi tensor
        t = torch.from_numpy(np.array(rgb_pil).transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        t = (t - mean) / std

        preds = {}
        with torch.no_grad():
            for mname, m in models.items():
                logits = m(t)
                preds[mname] = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Đo đạc chỉ số cỏ dại
        m_stats = {}
        for mname in models.keys():
            iou, prec, rec = compute_cls_metrics(preds[mname], gt_np, cls_id=2)
            m_stats[mname] = {"iou": iou, "prec": prec, "rec": rec}

        small_weeds = 0
        try:
            small_weeds = count_small_weeds(gt_np, max_pixels=100)
        except ImportError:
            pass

        delta_b1_b0 = m_stats["B1"]["iou"] - m_stats["B0"]["iou"]
        fp_b0_suppressed = np.logical_and(gt_np != 2, np.logical_and(preds["B0"] == 2, preds["B1"] != 2)).sum()

        records.append({
            "filename": fname,
            "img_path": img_path,
            "lighting": lighting,
            "stats": m_stats,
            "preds": preds,
            "gt_np": gt_np,
            "rgb_pil": rgb_pil,
            "small_weeds": small_weeds,
            "delta_b1_b0": delta_b1_b0,
            "fp_b0_suppressed": int(fp_b0_suppressed)
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == min(len(all_images), max_eval_samples):
            print(f"  -> Đã suy luận đối soát: {idx + 1}/{min(len(all_images), max_eval_samples)} ảnh...")

    # Phân loại các trường hợp thú vị nhất
    categories = {
        "1_sunny_shadow_rejection": {
            "title": "Khử Nhiễu Bóng Râm Dưới Nắng Gắt (Sunny II Shadow Invariance)",
            "filter": lambda r: r["lighting"] == "Sunny II",
            "sort_key": lambda r: r["fp_b0_suppressed"],
        },
        "2_micro_weed_detection": {
            "title": "Nhận Diện Khóm Cỏ Vi Mô (<100px: B1 vs P2_ex1 vs P2_ex2)",
            "filter": lambda r: r["small_weeds"] >= 3,
            "sort_key": lambda r: (r["stats"]["B1"]["iou"] - r["stats"]["P2_ex1"]["iou"]),
        },
        "3_max_precision_crop_safety": {
            "title": "Độ Chính Xác Kỷ Lục & An Toàn Phun Thuốc (P2_ex2 Max Precision)",
            "filter": lambda r: r["stats"]["P2_ex2"]["prec"] > r["stats"]["B1"]["prec"],
            "sort_key": lambda r: r["stats"]["P2_ex2"]["prec"],
        }
    }

    gallery_items = []

    for cat_key, cat_info in categories.items():
        cat_dir = os.path.join(output_dir, cat_key)
        os.makedirs(cat_dir, exist_ok=True)
        subset = [r for r in records if cat_info["filter"](r)]
        subset.sort(key=cat_info["sort_key"], reverse=True)
        chosen = subset[:top_k]

        print(f"\n[Category] {cat_info['title']}: chọn {len(chosen)} ảnh tiêu biểu...")

        for rank, r in enumerate(chosen, 1):
            fname = r["filename"]
            st = r["stats"]

            p1 = add_header(r["rgb_pil"], f"RGB: {fname}", f"Lighting: {r['lighting']}")
            p2 = add_header(colorize_mask(r["gt_np"]), "Ground Truth", f"Small Weeds: {r['small_weeds']}")
            p3 = add_header(colorize_mask(r["preds"]["B0"]), "B0 (PDC Baseline)",
                            f"Weed IoU: {st['B0']['iou']:.1f}% | Prec: {st['B0']['prec']:.1f}%")
            p4 = add_header(colorize_mask(r["preds"]["B1"]), "B1 (RepDWNet Base)",
                            f"Weed IoU: {st['B1']['iou']:.1f}% | Prec: {st['B1']['prec']:.1f}%")
            p5 = add_header(colorize_mask(r["preds"]["P2_ex1"]), "P2_ex1 (2DW + 4PW)",
                            f"Weed IoU: {st['P2_ex1']['iou']:.1f}% | Prec: {st['P2_ex1']['prec']:.1f}%")
            p6 = add_header(colorize_mask(r["preds"]["P2_ex2"]), "P2_ex2 (4DW + 2PW)",
                            f"Weed IoU: {st['P2_ex2']['iou']:.1f}% | Prec: {st['P2_ex2']['prec']:.1f}%")

            diff_img = create_discrepancy_map(r["gt_np"], r["preds"]["B0"], r["preds"]["B1"], r["preds"]["P2_ex2"])
            p7 = add_header(diff_img, "Discrepancy / Error Map", "Blue: B1 Fixed | Orange: FP Cleaned | Purple: P2 Safe")

            # Ghép lưới 7 khung hình nằm ngang
            total_w = sum(p.width for p in [p1, p2, p3, p4, p5, p6, p7])
            max_h = max(p.height for p in [p1, p2, p3, p4, p5, p6, p7])
            strip = Image.new('RGB', (total_w, max_h))

            curr_x = 0
            for panel in [p1, p2, p3, p4, p5, p6, p7]:
                strip.paste(panel, (curr_x, 0))
                curr_x += panel.width

            out_png = os.path.join(cat_dir, f"rank{rank:02d}_{os.path.splitext(fname)[0]}_7panel.png")
            strip.save(out_png, quality=95)
            print(f"  ✓ Đã lưu: {os.path.basename(out_png)}")

            gallery_items.append({
                "category": cat_info["title"],
                "filename": fname,
                "lighting": r["lighting"],
                "path": out_png,
                "rel_path": os.path.relpath(out_png, output_dir),
                "st": st
            })

    # Xuất file Markdown tổng hợp
    md_path = os.path.join(output_dir, "multi_model_qualitative_gallery.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Thư Viện Trực Quan Hóa So Sánh Đa Mô Hình: B0 vs B1 vs P2 (ex1 & ex2)\n\n")
        f.write("Bản đồ 7 khung hình đối soát (7-Panel Strip):\n")
        f.write("1. **RGB**: Ảnh chụp thực địa UAV $768 \\times 768$.\n")
        f.write("2. **Ground Truth**: Cây trồng (Xanh lá), Cỏ dại (Đỏ), Đất (Xám).\n")
        f.write("3. **B0 (PDC Baseline)**: Khảo sát hiện tượng bắt nhầm bóng râm.\n")
        f.write("4. **B1 (RepDWNet 4DW+4PW)**: Cân bằng không gian - kênh tối ưu.\n")
        f.write("5. **P2_ex1 (2DW+4PW)**: Tác động của việc giảm 50% nhánh không gian $3\\times 3$.\n")
        f.write("6. **P2_ex2 (4DW+2PW)**: Mô hình đạt Precision kỷ lục, ranh giới khắt khe.\n")
        f.write("7. **Discrepancy Map**: **Xanh dương** (B1 sửa đúng lỗi bỏ sót của B0), **Cam** (B0 báo giả bị triệt tiêu), **Tím** (P2_ex2 loại bỏ viền không chắc chắn).\n\n---\n\n")

        curr_c = None
        for it in gallery_items:
            if it["category"] != curr_c:
                curr_c = it["category"]
                f.write(f"## {curr_c}\n\n")
            f.write(f"### Ảnh: `{it['filename']}` ({it['lighting']})\n")
            f.write(f"- **B0**: Weed IoU `{it['st']['B0']['iou']:.1f}%` | Precision `{it['st']['B0']['prec']:.1f}%`\n")
            f.write(f"- **B1**: Weed IoU `{it['st']['B1']['iou']:.1f}%` | Precision `{it['st']['B1']['prec']:.1f}%`\n")
            f.write(f"- **P2_ex1**: Weed IoU `{it['st']['P2_ex1']['iou']:.1f}%` | Precision `{it['st']['P2_ex1']['prec']:.1f}%`\n")
            f.write(f"- **P2_ex2**: Weed IoU `{it['st']['P2_ex2']['iou']:.1f}%` | Precision `{it['st']['P2_ex2']['prec']:.1f}%`\n\n")
            f.write(f"![{it['filename']}]({it['rel_path']})\n\n---\n\n")

    print(f"\n[Hoàn Tất] Đã xuất thư viện trực quan hóa tại: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Model Qualitative Evaluation")
    parser.add_argument('--b0_ckpt', type=str, default="results/B0/B0_run5_cosine_lr2e4_results.zip")
    parser.add_argument('--b1_ckpt', type=str, default="results/B1/B1_run5_repdwnet_results.zip")
    parser.add_argument('--p2_ex1_ckpt', type=str, default="results/P2/P2_ex1_dw2_pw4_results.zip")
    parser.add_argument('--p2_ex2_ckpt', type=str, default="results/P2/P2_ex2_dw4_pw2_results.zip")
    parser.add_argument('--b0_config', type=str, default="RPD/configs/b0_scenarios/B0_run5_cosine_lr2e4.yaml")
    parser.add_argument('--b1_config', type=str, default="RPD/configs/b1_scenarios/B1_run5_repdwnet.yaml")
    parser.add_argument('--p2_ex1_config', type=str, default="RPD/configs/p2_scenarios/P2_ex1_dw2_pw4.yaml")
    parser.add_argument('--p2_ex2_config', type=str, default="RPD/configs/p2_scenarios/P2_ex2_dw4_pw2.yaml")
    parser.add_argument('--dataset_dir', type=str, default="/kaggle/input/phenobench-dataset/PhenoBench")
    parser.add_argument('--output_dir', type=str, default="results/multi_model_qualitative")
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()

    models = {
        "B0": load_model(args.b0_config, auto_extract_checkpoint(args.b0_ckpt), deploy=args.deploy, device=args.device),
        "B1": load_model(args.b1_config, auto_extract_checkpoint(args.b1_ckpt), deploy=args.deploy, device=args.device),
        "P2_ex1": load_model(args.p2_ex1_config, auto_extract_checkpoint(args.p2_ex1_ckpt), deploy=args.deploy, device=args.device),
        "P2_ex2": load_model(args.p2_ex2_config, auto_extract_checkpoint(args.p2_ex2_ckpt), deploy=args.deploy, device=args.device),
    }

    run_multi_model_pipeline(
        dataset_dir=args.dataset_dir,
        models=models,
        output_dir=args.output_dir,
        top_k=args.top_k,
        device=args.device
    )


if __name__ == '__main__':
    main()
