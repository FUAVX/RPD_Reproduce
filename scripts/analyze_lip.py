#!/usr/bin/env python3
"""Offline LIP importance analysis on native validation images (no train hooks).

python scripts/analyze_lip.py --config configs/p3_scenarios/P3_LIP_Ex1_Early.yaml \
  --checkpoint RUN/training_output/checkpoints/last.ckpt --dataset-dir DATA \
  --output-dir RUN/lip_analysis --device cuda --max-images 20
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F
import yaml
from models.rpdnet.lip import BottleneckLIP


def window_view(x):
    """N,C,H,W -> N,C,H//2,W//2,4, TL/TR/BL/BR; floor odd edges."""
    n, c, h, w = x.shape
    h, w = h // 2, w // 2
    if not h or not w:
        raise ValueError('A complete 2x2 window is required.')
    return (x[:, :, :2*h, :2*w].reshape(n, c, h, 2, w, 2)
            .permute(0, 1, 2, 4, 3, 5).reshape(n, c, h, w, 4))


def analyze_stage(logits, mask, stage):
    """One post-gate tensor and post-preprocessing GT; return per-window metrics."""
    if mask.ndim == 2:
        mask = mask[None]
    if logits.shape[0] != mask.shape[0] or stage not in (1, 2, 3, 4):
        raise ValueError('Batch/stage mismatch.')
    t = 2 ** (stage - 1)
    if tuple(logits.shape[-2:]) != (mask.shape[-2] // t, mask.shape[-1] // t):
        raise ValueError('GT and stage grid do not align; do not resize GT adaptively.')
    # CPU float32, bounded post-gate logits; no graph retained.
    z = logits.detach().float().cpu()
    gt = mask.detach().cpu()
    if not torch.isfinite(z).all() or not torch.all((gt == 0) | (gt == 1) | (gt == 2) | (gt == 255)):
        raise ValueError('Non-finite logits or unexpected remapped semantic labels.')
    g = F.avg_pool2d((gt == 2).float()[:, None], t, t)
    valid = F.avg_pool2d((gt != 255).float()[:, None], t, t) == 1
    g = window_view(g)[:, 0]
    valid = window_view(valid.float())[:, 0].bool().all(-1)
    a = g.mean(-1)
    mixed = valid & (a > 0) & (a < 1)
    equal = (g == g[..., :1]).all(-1)
    weights = window_view(z).softmax(-1)
    weighted_weed = (weights * g[:, None]).sum(-1)
    # Divide only selected mixed windows, never divide by zero then mask NaNs.
    enrichment = weighted_weed.permute(0, 2, 3, 1)[mixed] / a[mixed].unsqueeze(-1)
    entropy = (-torch.xlogy(weights, weights).sum(-1) / math.log(4))
    entropy = entropy.permute(0, 2, 3, 1)[mixed]
    return {
        'enrichment': enrichment.mean(-1), 'entropy': entropy.mean(-1),
        'occupancy': a[mixed], 'equal_occupancy': equal[mixed],
        'total_windows': valid.numel(), 'invalid_windows': int((~valid).sum()),
        'mixed_windows': int(mixed.sum()), 'valid_windows': int(valid.sum()),
    }


def describe(values):
    if not values.numel():
        return None
    return {'mean': float(values.mean()), 'min': float(values.min()),
            'p25': float(values.quantile(.25)), 'median': float(values.median()),
            'p75': float(values.quantile(.75)), 'max': float(values.max())}


def summarize_windows(metrics):
    result = {key: metrics[key] for key in
              ('total_windows', 'invalid_windows', 'valid_windows', 'mixed_windows')}
    for group, select in [('all_mixed', torch.ones_like(metrics['equal_occupancy'])),
                           ('equal_occupancy', metrics['equal_occupancy']),
                           ('varying_occupancy', ~metrics['equal_occupancy'])]:
        result[group] = {'windows': int(select.sum()),
                         **{key: describe(metrics[key][select]) for key in
                            ('enrichment', 'entropy', 'occupancy')}}
    return result


class LIPCapture:
    """Context-managed, detached gate hooks; no output replacement or extra forward."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.logits = {}

    def __enter__(self):
        for stage in range(1, 5):
            pool = getattr(self.model, f'down{stage}')[0]
            if isinstance(pool, BottleneckLIP):
                def capture(module, inputs, output, stage=stage):
                    self.logits[stage] = output.detach().float().cpu().clone()
                self.handles.append(pool.logit.gate.register_forward_hook(capture))
        if not self.handles:
            raise ValueError('Model contains no LIP stages.')
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.logits.clear()


def load_checkpoint(model, checkpoint):
    """Read trusted project checkpoints; support Lightning network.* and raw states."""
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw)
    if any(key.startswith('network.') for key in state):
        state = {key[len('network.'):]: value for key, value in state.items()
                 if key.startswith('network.')}
    elif state and all(key.startswith('model.') for key in state):
        state = {key[len('model.'):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def aggregate_images(rows):
    """Equal image weighting; images without mixed windows are reported, not zeroed."""
    summary = {}
    for stage in sorted({row['stage'] for row in rows}):
        selected = [row for row in rows if row['stage'] == stage]
        entry = {'images': len(selected),
                 'images_without_mixed_windows': sum(not row['mixed_windows'] for row in selected),
                 'invalid_windows': sum(row['invalid_windows'] for row in selected),
                 'mixed_windows': sum(row['mixed_windows'] for row in selected)}
        for group in ('all_mixed', 'equal_occupancy', 'varying_occupancy'):
            eligible = [row[group] for row in selected if row[group]['windows']]
            entry[group] = {'eligible_images': len(eligible),
                            'windows': sum(row[group]['windows'] for row in selected)}
            for metric in ('enrichment', 'entropy', 'occupancy'):
                entry[group][metric+'_image_mean'] = (sum(row[metric]['mean'] for row in eligible)
                                                      / len(eligible) if eligible else None)
        summary[str(stage)] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--save-windows', action='store_true', help='Save per-window E/H/A tensors on CPU.')
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error('--max-images must be positive')
    from models import get_backbone
    from datasets import get_data_module
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['data']['path_to_dataset'] = args.dataset_dir
    dm = get_data_module(cfg)
    dm.setup('validate')
    model = get_backbone(cfg).to(args.device).eval()
    load_checkpoint(model, args.checkpoint)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    count = min(len(dm.val_ds), args.max_images) if args.max_images else len(dm.val_ds)
    with LIPCapture(model) as capture, torch.inference_mode():
        for i in range(count):
            # get_val_item is the existing P2 native-resolution evaluation path.
            # No new crop/normalization or stochastic transforms are introduced here.
            sample = dm.val_ds[i]
            capture.logits.clear()
            model(sample['input_image'][None].to(args.device))
            for stage, z in capture.logits.items():
                metrics = analyze_stage(z, sample['anno'], stage)
                row = {'image': sample['fname'], 'stage': stage, **summarize_windows(metrics)}
                rows.append(row)
                if args.save_windows:
                    torch.save({key: metrics[key] for key in
                                ('enrichment', 'entropy', 'occupancy', 'equal_occupancy')},
                               output / f'{i:05d}_down{stage}_windows.pt')
    metadata = {'config': str(Path(args.config).resolve()), 'checkpoint': str(Path(args.checkpoint).resolve()),
                'dataset': args.dataset_dir, 'images': count, 'device': args.device,
                'input_policy': 'existing P2 val_ds native image and remapped mask, no extra geometry',
                'logits': 'post 12*sigmoid, before exp', 'pooling': 'k2/s2/p0',
                'ownership': 't=2**(stage-1), origin=(0,0), full-valid windows only',
                'aggregation': 'mean channels -> mean mixed windows per image -> equal image mean',
                'equal_occupancy': 'exact equality, no threshold', 'analysis_epsilon': None,
                'summary': aggregate_images(rows)}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    (output / 'per_image.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    with (output / 'per_image.csv').open('w', newline='') as stream:
        fields = ['image', 'stage', 'total_windows', 'invalid_windows', 'mixed_windows', 'E_mean', 'H_mean']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            group = row['all_mixed']
            writer.writerow({**{key: row[key] for key in fields[:5]},
                             'E_mean': group['enrichment']['mean'] if group['enrichment'] else None,
                             'H_mean': group['entropy']['mean'] if group['entropy'] else None})
    print(json.dumps(metadata['summary'], indent=2))


if __name__ == '__main__':
    main()
