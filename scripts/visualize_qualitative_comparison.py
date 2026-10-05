#!/usr/bin/env python3
"""
Qualitative Segmentation Comparison Generator: B0 (RPDNet) vs B1 (RepDWNet)
=============================================================================
Trực quan hóa so sánh chất lượng phân đoạn trên từng ảnh thực tế:
- Tự động quét tập Validation của PhenoBench (769 ảnh).
- Phân nhóm theo điều kiện ánh sáng (Sunny I, Sunny II, Overcast).
- Tự động phát hiện các ca tiêu biểu B1 vượt trội B0 (nắng gắt, bóng đổ, cỏ li ti, khử báo giả).
- Xuất dải ảnh 5 khung hình (5-Panel Strip) độ phân giải cao kèm bảng chú thích chi tiết:
  [RGB Input] - [Ground Truth] - [B0 Pred] - [B1 Pred] - [Error / Improvement Map]

Usage:
    python scripts/visualize_qualitative_comparison.py \
        --b0_ckpt results/B0/checkpoints/best.ckpt \
        --b1_ckpt results/B1/checkpoints/best.ckpt \
        --dataset_dir /kaggle/input/phenobench-dataset/PhenoBench \
        --output_dir results/qualitative_comparison \
        --top_k 5 \
        --device cuda
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

# Thêm path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RPD_DIR = os.path.join(BASE_DIR, 'RPD')
if RPD_DIR not in sys.path:
    sys.path.insert(0, RPD_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models import get_backbone
from models.rpdnet.RPD_Module import RPD_model_deploy

# Bảng màu chuẩn PhenoBench
# 0: Soil/Background (Đen/Xám đậm)
# 1: Crop (Xanh lá tươi)
# 2: Weed (Đỏ tươi)
COLOR_MAP = {
    0: (35, 35, 35),       # Background
    1: (46, 204, 113),     # Crop: #2ecc71
    2: (231, 76, 60),      # Weed: #e74c3c
}

# Màu cho bản đồ cải thiện (Difference Map)
COLOR_DIFF = {
    'correct_both': (40, 40, 40),       # Cả 2 đều đúng (nền mờ)
    'b1_corrected': (52, 152, 219),     # Xanh dương: B1 sửa đúng lỗi của B0!
    'b0_false_alarm': (230, 126, 34),   # Cam: B0 báo giả (False Positive) bị B1 khử sạch
    'both_error': (192, 57, 43),        # Đỏ đậm: Vẫn còn lỗi
    'crop_true': (39, 174, 96),         # Cây trồng đúng
}


def auto_extract_checkpoint(ckpt_or_zip_path, target_pattern="best_mIoU"):
    """Nếu truyền vào file zip, tự động giải nén checkpoint phù hợp nhất."""
    if not os.path.exists(ckpt_or_zip_path):
        raise FileNotFoundError(f"Không tìm thấy file: {ckpt_or_zip_path}")

    if ckpt_or_zip_path.endswith('.ckpt') or ckpt_or_zip_path.endswith('.pth'):
        return ckpt_or_zip_path

    if ckpt_or_zip_path.endswith('.zip'):
        extract_dir = os.path.join(os.path.dirname(ckpt_or_zip_path), 'extracted_ckpts')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(ckpt_or_zip_path, 'r') as zf:
            matching_files = [f for f in zf.namelist() if target_pattern in f and f.endswith('.ckpt')]
            if not matching_files:
                matching_files = [f for f in zf.namelist() if f.endswith('.ckpt')]
            if not matching_files:
                raise ValueError(f"Không tìm thấy file .ckpt nào trong {ckpt_or_zip_path}")
            
            chosen_file = matching_files[0]
            target_path = os.path.join(extract_dir, os.path.basename(chosen_file))
            if not os.path.exists(target_path):
                print(f"[Auto-Extract] Đang giải nén {chosen_file} -> {target_path}")
                with open(target_path, 'wb') as f_out:
                    f_out.write(zf.read(chosen_file))
            return target_path

    raise ValueError(f"Định dạng không hỗ trợ: {ckpt_or_zip_path}")


def load_model_from_checkpoint(config_path, ckpt_path, deploy=False, device='cuda'):
    """Tải và khởi tạo mô hình."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cfg['backbone']['deploy'] = False
    model = get_backbone(cfg)

    print(f"  [Load Model] Nạp checkpoint: {os.path.basename(ckpt_path)}")
    raw_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict = raw_dict.get('state_dict', raw_dict)
    cleaned = { (k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items() }
    model.load_state_dict(cleaned, strict=False)

    if deploy:
        print("  [Deploy Mode] Fuse sang single-branch...")
        model = RPD_model_deploy(model, do_copy=True)

    model = model.to(device)
    model.eval()
    return model


def find_dataset_dir(user_dir=None):
    """Tự động phát hiện thư mục PhenoBench."""
    candidates = [
        user_dir,
        "/kaggle/input/phenobench-dataset/PhenoBench",
        "/kaggle/input/datasets/ndhoang2310/phenobench-dataset/PhenoBench",
        "/kaggle/input/phenobench-dataset",
        "./data/PhenoBench",
        "../data/PhenoBench"
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, 'val', 'images')):
            return c
    return None


def colorize_mask(mask_np):
    """Chuyển mask số nguyên (0, 1, 2) thành ảnh màu RGB."""
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, rgb in COLOR_MAP.items():
        color_img[mask_np == cls_id] = rgb
    return Image.fromarray(color_img)


def create_difference_map(rgb_np, gt_np, b0_np, b1_np):
    """
    Tạo bản đồ khác biệt (Error / Improvement Map):
    - Xanh dương: B1 dự đoán đúng nơi B0 bị sai (Cải thiện rõ nét)
    - Cam: B0 báo giả Weed bị B1 khử sạch
    - Vàng/Đỏ: Lỗi còn lại
    """
    h, w = gt_np.shape
    diff_rgb = np.array(rgb_np, dtype=np.float32) * 0.35  # Làm mờ ảnh RGB nền

    # 1. B1 sửa đúng lỗi của B0 trên lớp cỏ dại (Weed)
    b1_fixed_weed = (gt_np == 2) & (b1_np == 2) & (b0_np != 2)
    diff_rgb[b1_fixed_weed] = [52, 152, 219]  # Xanh dương sáng (#3498db)

    # 2. B0 báo nhầm là cỏ dại (False Positive), B1 khử đúng
    b0_fp_weed_fixed = (gt_np != 2) & (b0_np == 2) & (b1_np == gt_np)
    diff_rgb[b0_fp_weed_fixed] = [243, 156, 18]  # Cam vàng (#f39c12)

    # 3. Cả 2 đều nhận diện đúng cỏ dại
    both_correct_weed = (gt_np == 2) & (b1_np == 2) & (b0_np == 2)
    diff_rgb[both_correct_weed] = [231, 76, 60]  # Đỏ tươi (#e74c3c)

    # 4. Cả 2 đều nhận diện đúng cây trồng
    both_correct_crop = (gt_np == 1) & (b1_np == 1) & (b0_np == 1)
    diff_rgb[both_correct_crop] = [39, 174, 96]  # Xanh lá đậm

    return Image.fromarray(diff_rgb.astype(np.uint8))


def overlay_mask_on_rgb(rgb_pil, mask_np, alpha=0.55):
    """Phủ mask bán trong suốt lên ảnh RGB."""
    mask_rgb = colorize_mask(mask_np)
    blended = Image.blend(rgb_pil, mask_rgb, alpha=alpha)
    return blended


def add_header_to_panel(panel_img, title, subtitle=None, bg_color=(25, 25, 25), text_color=(255, 255, 255)):
    """Thêm tiêu đề và thông số lên phía trên khung hình."""
    header_h = 55 if subtitle else 40
    new_w, new_h = panel_img.width, panel_img.height + header_h
    new_img = Image.new('RGB', (new_w, new_h), color=bg_color)
    draw = ImageDraw.Draw(new_img)

    # Viết tiêu đề chính
    draw.text((12, 8), title, fill=text_color)
    if subtitle:
        draw.text((12, 28), subtitle, fill=(200, 200, 200))

    new_img.paste(panel_img, (0, header_h))
    return new_img


def compute_metrics(pred_np, gt_np):
    """Tính toán IoU, Precision, Recall cho các lớp."""
    metrics = {}
    for cls_id, name in [(1, 'crop'), (2, 'weed')]:
        tp = np.logical_and(pred_np == cls_id, gt_np == cls_id).sum()
        fp = np.logical_and(pred_np == cls_id, gt_np != cls_id).sum()
        fn = np.logical_and(pred_np != cls_id, gt_np == cls_id).sum()

        iou = (tp / (tp + fp + fn)) * 100.0 if (tp + fp + fn) > 0 else 0.0
        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0

        metrics[f'{name}_iou'] = iou
        metrics[f'{name}_prec'] = prec
        metrics[f'{name}_rec'] = rec

    # Background IoU
    tp_bg = np.logical_and(pred_np == 0, gt_np == 0).sum()
    fp_bg = np.logical_and(pred_np == 0, gt_np != 0).sum()
    fn_bg = np.logical_and(pred_np != 0, gt_np == 0).sum()
    bg_iou = (tp_bg / (tp_bg + fp_bg + fn_bg)) * 100.0 if (tp_bg + fp_bg + fn_bg) > 0 else 0.0

    metrics['miou'] = (bg_iou + metrics['crop_iou'] + metrics['weed_iou']) / 3.0
    return metrics


def run_qualitative_pipeline(dataset_dir, b0_model, b1_model, output_dir,
                             top_k=5, device='cuda', max_eval_samples=769):
    """Chạy toàn bộ quy trình đánh giá và sinh ảnh trực quan."""
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(dataset_dir, 'val', 'images')
    semantics_dir = os.path.join(dataset_dir, 'val', 'semantics')

    all_images = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    if not all_images:
        raise FileNotFoundError(f"Không tìm thấy ảnh PNG trong {images_dir}")

    print(f"\n[Dataset] Tìm thấy tổng cộng {len(all_images)} ảnh trong val set.")
    print(f"[Evaluation] Bắt đầu suy luận so sánh song song B0 vs B1 trên {min(len(all_images), max_eval_samples)} ảnh...\n")

    records = []

    for idx, img_path in enumerate(all_images[:max_eval_samples]):
        base_name = os.path.basename(img_path)
        sem_path = os.path.join(semantics_dir, base_name)
        if not os.path.exists(sem_path):
            continue

        # Phân loại ánh sáng theo prefix ngày chụp
        if base_name.startswith('05-15'):
            lighting = 'Sunny I'
        elif base_name.startswith('05-26'):
            lighting = 'Sunny II'
        elif base_name.startswith('06-05'):
            lighting = 'Overcast'
        else:
            lighting = 'Unknown'

        # Đọc ảnh RGB và GT mask
        rgb_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = rgb_pil.size
        # Đảm bảo resize về 768x768 nếu cần
        if (orig_w, orig_h) != (768, 768):
            rgb_resized = rgb_pil.resize((768, 768), Image.BILINEAR)
        else:
            rgb_resized = rgb_pil

        gt_pil = Image.open(sem_path)
        if (orig_w, orig_h) != (768, 768):
            gt_pil = gt_pil.resize((768, 768), Image.NEAREST)
        gt_np = np.array(gt_pil)

        # Chuyển đổi tensor chuẩn hóa
        img_tensor = torch.from_numpy(np.array(rgb_resized)).permute(2, 0, 1).float() / 255.0
        # Normalization chuẩn ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        input_tensor = img_tensor.unsqueeze(0).to(device)

        # Suy luận B0 và B1
        with torch.no_grad():
            out_b0 = b0_model(input_tensor)
            out_b1 = b1_model(input_tensor)

            pred_b0 = out_b0.argmax(dim=1).squeeze(0).cpu().numpy()
            pred_b1 = out_b1.argmax(dim=1).squeeze(0).cpu().numpy()

        m_b0 = compute_metrics(pred_b0, gt_np)
        m_b1 = compute_metrics(pred_b1, gt_np)

        delta_weed_iou = m_b1['weed_iou'] - m_b0['weed_iou']
        delta_prec = m_b1['weed_prec'] - m_b0['weed_prec']
        weed_pixel_pct = (gt_np == 2).sum() / (768 * 768) * 100.0

        records.append({
            'filename': base_name,
            'lighting': lighting,
            'rgb_pil': rgb_resized,
            'gt_np': gt_np,
            'pred_b0': pred_b0,
            'pred_b1': pred_b1,
            'm_b0': m_b0,
            'm_b1': m_b1,
            'delta_weed_iou': delta_weed_iou,
            'delta_prec': delta_prec,
            'weed_pixel_pct': weed_pixel_pct
        })

        if (idx + 1) % 50 == 0:
            print(f"  * Đã suy luận: {idx+1}/{min(len(all_images), max_eval_samples)} ảnh...")

    print("\n[Analysis] Hoàn tất suy luận! Tiến hành phân loại & trích xuất các ca phân đoạn tiêu biểu...")

    # Tiêu chí chọn ca:
    # 1. Sunny II Top IoU Delta
    sunny_ii_top = sorted([r for r in records if r['lighting'] == 'Sunny II' and r['weed_pixel_pct'] > 0.5],
                          key=lambda x: x['delta_weed_iou'], reverse=True)[:top_k]

    # 2. Cỏ dại nhỏ li ti (Micro-weeds: weed_pixel_pct < 2.0% nhưng B1 nhận diện tốt)
    small_weed_top = sorted([r for r in records if 0.2 < r['weed_pixel_pct'] < 2.5],
                            key=lambda x: x['delta_weed_iou'], reverse=True)[:top_k]

    # 3. Khử báo giả vượt trội (Delta Precision lớn nhất)
    prec_boost_top = sorted([r for r in records if r['m_b0']['weed_prec'] < 65.0 and r['weed_pixel_pct'] > 0.5],
                            key=lambda x: x['delta_prec'], reverse=True)[:top_k]

    # 4. Overcast Top IoU Delta
    overcast_top = sorted([r for r in records if r['lighting'] == 'Overcast' and r['weed_pixel_pct'] > 0.5],
                          key=lambda x: x['delta_weed_iou'], reverse=True)[:top_k]

    categories = [
        ('sunny_ii_chói_sáng_bóng_đổ', 'Sunny II (Bóng đổ & Chói lóa gắt)', sunny_ii_top),
        ('cỏ_dại_li_ti_mới_nhú', 'Cỏ Dại Kích Thước Siêu Nhỏ (Micro-weeds)', small_weed_top),
        ('khử_báo_giả_tán_lá', 'Khử Báo Giả Tán Lá Cây Trồng (Weed Precision Boost)', prec_boost_top),
        ('overcast_trời_râm', 'Overcast (Ánh sáng tán xạ đồng đều)', overcast_top),
    ]

    saved_images_meta = []

    for slug, cat_title, items in categories:
        cat_dir = os.path.join(output_dir, slug)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n>>> Đang xuất ảnh cho nhóm: {cat_title} ({len(items)} ảnh)...")

        for rank, item in enumerate(items, 1):
            fname = item['filename']
            rgb_pil = item['rgb_pil']
            gt_np = item['gt_np']
            pred_b0 = item['pred_b0']
            pred_b1 = item['pred_b1']
            mb0 = item['m_b0']
            mb1 = item['m_b1']

            # Tạo 5 khung hình
            # Panel 1: RGB
            p1 = add_header_to_panel(rgb_pil, "1. Ảnh RGB Gốc (UAV)", f"{fname} | {item['lighting']}")

            # Panel 2: Ground Truth
            gt_overlay = overlay_mask_on_rgb(rgb_pil, gt_np)
            p2 = add_header_to_panel(gt_overlay, "2. Ground Truth Chuẩn",
                                      f"Xanh: Cây trồng | Đỏ: Cỏ dại ({item['weed_pixel_pct']:.1f}% DT)")

            # Panel 3: B0 Prediction
            b0_overlay = overlay_mask_on_rgb(rgb_pil, pred_b0)
            p3 = add_header_to_panel(b0_overlay, "3. B0: RPDNet Baseline (PDC)",
                                      f"Weed IoU: {mb0['weed_iou']:.1f}% | Prec: {mb0['weed_prec']:.1f}% | Rec: {mb0['weed_rec']:.1f}%",
                                      bg_color=(50, 20, 20))

            # Panel 4: B1 Prediction
            b1_overlay = overlay_mask_on_rgb(rgb_pil, pred_b1)
            delta_str = f"+{item['delta_weed_iou']:.1f}%" if item['delta_weed_iou'] >= 0 else f"{item['delta_weed_iou']:.1f}%"
            p4 = add_header_to_panel(b1_overlay, "4. B1: RepDWNet (Đề xuất)",
                                      f"Weed IoU: {mb1['weed_iou']:.1f}% ({delta_str}) | Prec: {mb1['weed_prec']:.1f}%",
                                      bg_color=(20, 50, 25))

            # Panel 5: Error / Improvement Map
            diff_img = create_difference_map(np.array(rgb_pil), gt_np, pred_b0, pred_b1)
            p5 = add_header_to_panel(diff_img, "5. Bản Đồ Cải Thiện (Improvement)",
                                      "Xanh dương: B1 sửa đúng lỗi B0 | Cam: B1 khử báo giả",
                                      bg_color=(20, 35, 55))

            # Ghép ngang 5 panels
            total_w = p1.width * 5
            total_h = p1.height
            strip = Image.new('RGB', (total_w, total_h))
            strip.paste(p1, (0, 0))
            strip.paste(p2, (p1.width, 0))
            strip.paste(p3, (p1.width * 2, 0))
            strip.paste(p4, (p1.width * 3, 0))
            strip.paste(p5, (p1.width * 4, 0))

            out_path = os.path.join(cat_dir, f"rank{rank:02d}_{os.path.splitext(fname)[0]}_comparison.png")
            strip.save(out_path, quality=95)
            print(f"  ✓ Đã lưu: {os.path.basename(out_path)} (Weed IoU: B0={mb0['weed_iou']:.1f}% -> B1={mb1['weed_iou']:.1f}%)")

            saved_images_meta.append({
                'category': cat_title,
                'path': out_path,
                'rel_path': os.path.relpath(out_path, output_dir),
                'filename': fname,
                'b0_weed_iou': mb0['weed_iou'],
                'b1_weed_iou': mb1['weed_iou'],
                'b0_prec': mb0['weed_prec'],
                'b1_prec': mb1['weed_prec'],
                'delta': item['delta_weed_iou']
            })

    # Tạo file Markdown Gallery
    md_gallery_path = os.path.join(output_dir, "qualitative_gallery.md")
    with open(md_gallery_path, 'w', encoding='utf-8') as f_md:
        f_md.write("# Thư Viện Trực Quan Hóa So Sánh Phân Đoạn Thực Tế: B1 (RepDWNet) vs B0 (RPDNet)\n\n")
        f_md.write("Bản đồ chú giải dải ảnh 5 khung hình:\n")
        f_md.write("1. **Ảnh RGB:** Ảnh chụp UAV mặt đất độ phân giải $768 \\times 768$.\n")
        f_md.write("2. **Ground Truth:** Xanh lá: Cây trồng, Đỏ: Cỏ dại, Đen: Đất nền.\n")
        f_md.write("3. **B0 (PDC Baseline):** Mô hình tái lập từ bài báo IEEE TGRS 2026.\n")
        f_md.write("4. **B1 (RepDWNet):** Kiến trúc đề xuất loại bỏ vi sai PDC sang RepDW.\n")
        f_md.write("5. **Bản đồ Cải thiện:** **Màu xanh dương** thể hiện các pixel cỏ dại B1 nhận diện chính xác mà B0 bỏ sót; **Màu cam** thể hiện các vùng báo giả của B0 được B1 khử sạch.\n\n---\n\n")

        current_cat = None
        for meta in saved_images_meta:
            if meta['category'] != current_cat:
                current_cat = meta['category']
                f_md.write(f"## {current_cat}\n\n")

            f_md.write(f"### Ảnh: `{meta['filename']}` (Weed IoU: B0 `{meta['b0_weed_iou']:.1f}%` $\\rightarrow$ B1 `{meta['b1_weed_iou']:.1f}%`, Delta: `+{meta['delta']:.1f}%`)\n")
            f_md.write(f"- **Weed Precision:** B0 `{meta['b0_prec']:.1f}%` $\\rightarrow$ B1 `{meta['b1_prec']:.1f}%`\n\n")
            f_md.write(f"![{meta['filename']}]({meta['rel_path']})\n\n---\n\n")

    print(f"\n[Done] Toàn bộ ảnh so sánh và thư viện Markdown đã được lưu tại: {output_dir}")
    print(f"       File mục lục: {md_gallery_path}")


def main():
    parser = argparse.ArgumentParser(description="Qualitative Visualizer B0 vs B1")
    parser.add_argument('--b0_config', type=str, default='RPD/configs/b0_scenarios/B0_run5_cosine_lr2e4.yaml')
    parser.add_argument('--b1_config', type=str, default='RPD/configs/b1_scenarios/B1_run5_repdwnet.yaml')
    parser.add_argument('--b0_ckpt', type=str, required=True, help="Path to B0 .ckpt or .zip")
    parser.add_argument('--b1_ckpt', type=str, required=True, help="Path to B1 .ckpt or .zip")
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='results/qualitative_comparison')
    parser.add_argument('--top_k', type=int, default=3, help="Số ảnh tiêu biểu mỗi nhóm")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_eval_samples', type=int, default=769)
    parser.add_argument('--deploy', action='store_true', help="Chạy ở trạng thái fused deploy")
    args = parser.parse_args()

    # Tìm dataset
    ds_dir = find_dataset_dir(args.dataset_dir)
    if not ds_dir:
        raise FileNotFoundError("Không tìm thấy thư mục PhenoBench! Vui lòng truyền --dataset_dir.")

    # Trích xuất / nạp checkpoint
    b0_ckpt_path = auto_extract_checkpoint(args.b0_ckpt, "best_mIoU")
    b1_ckpt_path = auto_extract_checkpoint(args.b1_ckpt, "best_mIoU")

    # Nạp mô hình
    print(f"[Init] Đang khởi tạo mô hình B0 và B1 trên {args.device}...")
    b0_model = load_model_from_checkpoint(args.b0_config, b0_ckpt_path, deploy=args.deploy, device=args.device)
    b1_model = load_model_from_checkpoint(args.b1_config, b1_ckpt_path, deploy=args.deploy, device=args.device)

    # Chạy pipeline
    run_qualitative_pipeline(
        dataset_dir=ds_dir,
        b0_model=b0_model,
        b1_model=b1_model,
        output_dir=args.output_dir,
        top_k=args.top_k,
        device=args.device,
        max_eval_samples=args.max_eval_samples
    )


if __name__ == '__main__':
    main()
