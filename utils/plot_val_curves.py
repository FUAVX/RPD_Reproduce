"""
Automatic Validation Curve Generator and Summary Reporter for RPD-Net
Generates 4-panel visual plots and summary metrics from val_history.csv or evaluation YAMLs.
"""

import os
import sys
import glob
import argparse
import pandas as pd


def load_val_data(export_dir: str) -> pd.DataFrame:
    """Load validation metrics from val_history.csv or reconstruct from evaluation directory."""
    csv_path = os.path.join(export_dir, 'val_history.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[Notice] Could not read {csv_path}: {e}")

    # Fallback: check lightning_logs metrics.csv
    lightning_csvs = glob.glob(os.path.join(export_dir, '**', 'metrics.csv'), recursive=True)
    if lightning_csvs:
        try:
            df = pd.read_csv(lightning_csvs[0])
            if 'val_mIoU' in df.columns:
                val_df = df.dropna(subset=['val_mIoU']).copy()
                if not val_df.empty:
                    return val_df
        except Exception:
            pass

    return pd.DataFrame()


def generate_curves_and_summary(export_dir: str, output_image_name: str = 'val_curves.png') -> bool:
    """Generate 2x2 subplot curves and print summary statistics."""
    df = load_val_data(export_dir)
    if df.empty:
        print(f"[Notice] No validation history found in {export_dir} to plot.")
        return False

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib not installed; skipping plot generation.")
        return False

    epochs = df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Validation Progression Report: {os.path.basename(os.path.normpath(export_dir))}', fontsize=14, fontweight='bold')

    # Panel 1: Loss Curves
    ax_loss = axes[0, 0]
    if 'train_loss' in df.columns and df['train_loss'].notna().any():
        ax_loss.plot(epochs, df['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        ax_loss.plot(epochs, df['val_loss'], label='Val Loss', color='#d62728', linewidth=2, linestyle='--')
    ax_loss.set_title('Loss Trajectory (Train vs Val)', fontweight='bold')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.grid(True, linestyle=':', alpha=0.6)
    ax_loss.legend()

    # Panel 2: IoU per Class & mIoU
    ax_iou = axes[0, 1]
    if 'mIoU' in df.columns:
        ax_iou.plot(epochs, df['mIoU'], label='Mean IoU (mIoU)', color='#2ca02c', linewidth=2.5)
    if 'weed_iou' in df.columns:
        ax_iou.plot(epochs, df['weed_iou'], label='Weed IoU (Class 2)', color='#d62728', linewidth=2.5, marker='o', markersize=4)
    if 'crop_iou' in df.columns:
        ax_iou.plot(epochs, df['crop_iou'], label='Crop IoU (Class 1)', color='#ff7f0e', linewidth=1.5, linestyle='-.')
    if 'soil_iou' in df.columns:
        ax_iou.plot(epochs, df['soil_iou'], label='Soil IoU (Class 0)', color='#7f7f7f', linewidth=1.5, linestyle=':')
    ax_iou.set_title('Segmentation Overlap (Per-Class & Mean IoU)', fontweight='bold')
    ax_iou.set_xlabel('Epoch')
    ax_iou.set_ylabel('IoU (0.0 - 1.0)')
    ax_iou.set_ylim([0.0, 1.05])
    ax_iou.grid(True, linestyle=':', alpha=0.6)
    ax_iou.legend()

    # Panel 3: Detailed Weed Metrics
    ax_weed = axes[1, 0]
    if 'weed_iou' in df.columns:
        ax_weed.plot(epochs, df['weed_iou'], label='Weed IoU', color='#d62728', linewidth=2.5)
    if 'weed_precision' in df.columns:
        ax_weed.plot(epochs, df['weed_precision'], label='Weed Precision', color='#9467bd', linewidth=1.8, linestyle='--')
    if 'weed_recall' in df.columns:
        ax_weed.plot(epochs, df['weed_recall'], label='Weed Recall', color='#17becf', linewidth=1.8, linestyle='-.')
    if 'weed_f1' in df.columns:
        ax_weed.plot(epochs, df['weed_f1'], label='Weed F1', color='#e377c2', linewidth=2.0)
    ax_weed.set_title('Weed Detection Deep-Dive (IoU, P, R, F1)', fontweight='bold')
    ax_weed.set_xlabel('Epoch')
    ax_weed.set_ylabel('Metric Score (0.0 - 1.0)')
    ax_weed.set_ylim([0.0, 1.05])
    ax_weed.grid(True, linestyle=':', alpha=0.6)
    ax_weed.legend()

    # Panel 4: Learning Rate Schedule
    ax_lr = axes[1, 1]
    if 'lr' in df.columns:
        ax_lr.plot(epochs, df['lr'], label='Learning Rate', color='#8c564b', linewidth=2)
    ax_lr.set_title('Learning Rate Progression', fontweight='bold')
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.grid(True, linestyle=':', alpha=0.6)
    ax_lr.legend()

    plt.tight_layout()
    output_path = os.path.join(export_dir, output_image_name)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Visual Report] Saved validation curves to: {output_path}")

    # Generate Summary text
    generate_summary_report(df, export_dir)
    return True


def generate_summary_report(df: pd.DataFrame, export_dir: str):
    """Print and save a formatted summary of best metrics."""
    if df.empty or 'weed_iou' not in df.columns:
        return

    best_weed_idx = df['weed_iou'].idxmax()
    best_weed_row = df.loc[best_weed_idx]

    best_miou_idx = df['mIoU'].idxmax() if 'mIoU' in df.columns else best_weed_idx
    best_miou_row = df.loc[best_miou_idx]

    summary_lines = [
        "=" * 65,
        "          RPD-NET TRAINING VALIDATION SUMMARY REPORT",
        "=" * 65,
        f"Export Directory     : {export_dir}",
        f"Total Evaluated Epochs: {len(df)} checkpoints recorded",
        "-" * 65,
        f"★ Peak Weed IoU      : {best_weed_row['weed_iou']:.4f} (at Epoch {int(best_weed_row.get('epoch', 0))})",
        f"  - Weed Precision   : {best_weed_row.get('weed_precision', 0):.4f}",
        f"  - Weed Recall      : {best_weed_row.get('weed_recall', 0):.4f}",
        f"  - Weed F1 Score    : {best_weed_row.get('weed_f1', 0):.4f}",
        f"  - Overall mIoU     : {best_weed_row.get('mIoU', 0):.4f}",
        "-" * 65,
        f"★ Peak Overall mIoU  : {best_miou_row.get('mIoU', 0):.4f} (at Epoch {int(best_miou_row.get('epoch', 0))})",
        f"  - Soil IoU         : {best_miou_row.get('soil_iou', 0):.4f}",
        f"  - Crop IoU         : {best_miou_row.get('crop_iou', 0):.4f}",
        f"  - Weed IoU         : {best_miou_row.get('weed_iou', 0):.4f}",
        "=" * 65,
    ]
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_file_path = os.path.join(export_dir, 'val_summary.txt')
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
    except Exception:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot validation curves from RPD training history')
    parser.add_argument('--log_dir', default='results', help='Path to results directory containing val_history.csv')
    parser.add_argument('--out', default='val_curves.png', help='Output image filename')
    args = parser.parse_args()

    generate_curves_and_summary(args.log_dir, args.out)
