"""
Multi-Run Comparison and Ranking Tool for RPD-Net Baseline B0 Scenarios
Loads val_history.csv from multiple runs, overlays Weed IoU / mIoU curves, and ranks configurations.
"""

import os
import glob
import argparse
import pandas as pd


def compare_runs(run_dirs: list, output_dir: str = 'comparison_results'):
    os.makedirs(output_dir, exist_ok=True)

    records = []
    run_histories = {}

    for rdir in run_dirs:
        rdir = rdir.rstrip('/')
        run_name = os.path.basename(rdir)
        csv_path = os.path.join(rdir, 'val_history.csv')
        if not os.path.exists(csv_path):
            candidates = glob.glob(os.path.join(rdir, '**', 'val_history.csv'), recursive=True)
            if candidates:
                csv_path = candidates[0]
            else:
                print(f"[Warning] No val_history.csv found in {rdir}. Skipping.")
                continue

        try:
            df = pd.read_csv(csv_path)
            if df.empty or 'weed_iou' not in df.columns:
                continue

            run_histories[run_name] = df

            best_weed_idx = df['weed_iou'].idxmax()
            best_row = df.loc[best_weed_idx]

            records.append({
                'Run': run_name,
                'Best_Epoch': int(best_row.get('epoch', 0)),
                'Peak_Weed_IoU': float(best_row.get('weed_iou', 0.0)),
                'Peak_mIoU': float(df['mIoU'].max()) if 'mIoU' in df.columns else 0.0,
                'Weed_Precision': float(best_row.get('weed_precision', 0.0)),
                'Weed_Recall': float(best_row.get('weed_recall', 0.0)),
                'Weed_F1': float(best_row.get('weed_f1', 0.0)),
                'Final_Weed_IoU': float(df['weed_iou'].iloc[-1]),
                'Final_mIoU': float(df['mIoU'].iloc[-1]) if 'mIoU' in df.columns else 0.0,
            })
        except Exception as e:
            print(f"[Error] Failed to process {csv_path}: {e}")

    if not records:
        print("[Error] No valid runs found to compare.")
        return

    ranking_df = pd.DataFrame(records)
    ranking_df = ranking_df.sort_values(by='Peak_Weed_IoU', ascending=False).reset_index(drop=True)
    ranking_df.index += 1  # 1-based rank
    ranking_df.index.name = 'Rank'

    # Save summary table
    table_csv = os.path.join(output_dir, 'ranking_table.csv')
    ranking_df.to_csv(table_csv)

    print("\n" + "=" * 80)
    print("                    RPD-NET B0 SCENARIOS RANKING TABLE")
    print("=" * 80)
    print(ranking_df.to_string())
    print("=" * 80)

    # Save markdown summary
    md_path = os.path.join(output_dir, 'ranking_summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# RPD-Net B0 Scenarios Ranking Summary\n\n")
        f.write(ranking_df.to_markdown())
        f.write("\n\n### Recommendation for B0* Base Model:\n")
        best_run = ranking_df.iloc[0]['Run']
        best_iou = ranking_df.iloc[0]['Peak_Weed_IoU']
        f.write(f"- **Top Performer:** `{best_run}` with Peak Weed IoU = **{best_iou:.4f}**\n")

    # Plot overlay curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('B0 Scenarios Cross-Comparison (Validation Progression)', fontsize=14, fontweight='bold')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

        for i, (name, df) in enumerate(run_histories.items()):
            c = colors[i % len(colors)]
            epochs = df['epoch'].tolist() if 'epoch' in df.columns else list(range(len(df)))

            # Left: Weed IoU
            ax1.plot(epochs, df['weed_iou'], label=name, color=c, linewidth=2.0)

            # Right: mIoU
            if 'mIoU' in df.columns:
                ax2.plot(epochs, df['mIoU'], label=name, color=c, linewidth=2.0)

        ax1.set_title('Weed IoU Progression (Primary Target)', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Weed IoU (0.0 - 1.0)')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        ax2.set_title('Mean IoU (mIoU) Progression (Overall Quality)', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU (0.0 - 1.0)')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'compare_all_runs.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[Visual Comparison] Saved comparison plot to: {plot_path}")
    except Exception as e:
        print(f"[Warning] Failed to generate comparison plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare multiple RPD training runs')
    parser.add_argument('--run_dirs', nargs='+', required=True, help='Paths to export directories of the runs')
    parser.add_argument('--out_dir', default='comparison_results', help='Output directory for comparison artifacts')
    args = parser.parse_args()

    compare_runs(args.run_dirs, args.out_dir)
