#!/usr/bin/env python3
"""Offline WSC: normalize native1024 once, crop1008, compare physical overlap.

This is a project crop-origin stability metric, not a causal aliasing metric.
Uses the original trained eval graph even when deploy parity fails.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

OFFSETS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
CROP_SIZE = 1008
ORIGIN = (8, 8)


def remap_semantics(raw):
    gt = torch.as_tensor(raw).long().clone()
    if not torch.all((gt == 0) | (gt == 1) | (gt == 2) | (gt == 3) | (gt == 4) | (gt == 255)):
        raise ValueError('Unexpected semantic label.')
    gt[gt == 3] = 1
    gt[gt == 4] = 2
    return gt


def aligned_slices(offset, shape):
    """Offset is shifted crop origin minus base origin, not image-roll direction."""
    base, shifted = [], []
    for delta, size in zip(offset, shape):
        if abs(delta) >= size:
            raise ValueError('No physical overlap.')
        start, stop = max(0, delta), min(size, size+delta)
        base.append(slice(start, stop))
        shifted.append(slice(start-delta, stop-delta))
    return tuple(base), tuple(shifted)


def wsc_pair(base, shifted, base_gt, offset):
    if base.ndim != 2 or base.shape != shifted.shape or base.shape != base_gt.shape:
        raise ValueError('Probability maps/GT must have identical 2D shapes.')
    for p in (base, shifted):
        if not torch.isfinite(p).all() or not torch.all((p >= 0) & (p <= 1)):
            raise ValueError('WSC needs finite probabilities in [0,1].')
    if not torch.all((base_gt == 0) | (base_gt == 1) | (base_gt == 2) | (base_gt == 255)):
        raise ValueError('WSC needs remapped GT.')
    a, b = aligned_slices(offset, base.shape)
    roi = base_gt[a] == 2
    count = int(roi.sum())
    if not count:
        return {'weed_pixels': 0, 'wsc': None, 'mean_abs_probability_error': None,
                'status': 'skipped_no_valid_weed'}
    error = float((base[a][roi]-shifted[b][roi]).abs().double().mean())
    return {'weed_pixels': count, 'wsc': 1-error,
            'mean_abs_probability_error': error, 'status': 'evaluated'}


def evaluate_normalized_image(model, normalized, gt, device):
    """normalized is produced ONCE from the full image by the caller."""
    if normalized.shape != (3, 1024, 1024) or gt.shape != (1024, 1024):
        raise ValueError('WSC-Now requires native RGB/GT1024; no resizing or padding.')
    if model.training:
        raise ValueError('WSC requires model.eval().')
    row, col = ORIGIN
    views = {'base': (0, 0), **OFFSETS}
    probabilities = {}
    with torch.inference_mode():
        for name, (dy, dx) in views.items():
            x = normalized[:, row+dy:row+dy+CROP_SIZE, col+dx:col+dx+CROP_SIZE]
            logits = model(x[None].to(device))
            if logits.shape != (1, 3, CROP_SIZE, CROP_SIZE) or not torch.isfinite(logits).all():
                raise ValueError('Nonfinite logits or wrong WSC output shape.')
            probabilities[name] = logits.softmax(1)[0, 2].float().cpu().clone()
    base_gt = gt[row:row+CROP_SIZE, col:col+CROP_SIZE].cpu()
    return [{'direction': name, 'offset_row': dy, 'offset_col': dx,
             **wsc_pair(probabilities['base'], probabilities[name], base_gt, (dy, dx))}
            for name, (dy, dx) in OFFSETS.items()]


def aggregate_wsc(rows):
    by_image = {}
    for r in rows:
        by_image.setdefault(r['image'], []).append(r)
    per_image = []
    for name, group in by_image.items():
        valid = [r['wsc'] for r in group if r['wsc'] is not None]
        per_image.append({'image': name, 'valid_offsets': len(valid),
                          'wsc': statistics.mean(valid) if valid else None})
    values = [r['wsc'] for r in per_image if r['wsc'] is not None]
    skipped = sum(r['wsc'] is None for r in rows)
    summary = {'mean_wsc': statistics.mean(values) if values else None,
               'std_wsc_across_images': statistics.pstdev(values) if values else None,
               'std_ddof': 0, 'images_seen': len(by_image), 'evaluated_images': len(values),
               'skipped_images': len(by_image)-len(values), 'total_image_offset_pairs': len(rows),
               'skipped_image_offset_pairs': skipped,
               'skipped_pairs_percent': 100*skipped/len(rows) if rows else None,
               'per_direction': {}}
    for direction in OFFSETS:
        valid = [r['wsc'] for r in rows if r['direction'] == direction and r['wsc'] is not None]
        summary['per_direction'][direction] = {'mean_wsc': statistics.mean(valid) if valid else None,
                                                'evaluated_pairs': len(valid)}
    return per_image, summary


def load_checkpoint(model, checkpoint):
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw)
    if any(k.startswith('network.') for k in state):
        state = {k[len('network.'):]: v for k, v in state.items() if k.startswith('network.')}
    elif state and all(k.startswith('model.') for k in state):
        state = {k[len('model.'):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return raw


def run_analysis(config, checkpoint, dataset_dir, output_dir, device='cuda', max_images=None, filenames=None):
    import numpy as np
    from PIL import Image
    from torchvision.transforms.functional import to_tensor
    import yaml
    from models import get_backbone
    from datasets.augmentations_normalize import SingleImageNormalizer
    cfg = yaml.safe_load(Path(config).read_text()) if not isinstance(config, dict) else config
    if cfg['data']['image_normalizer']['name'] != 'single_image_normalizer':
        raise ValueError('WSC must retain the selected P2 SingleImageNormalizer.')
    if max_images is not None and max_images < 1:
        raise ValueError('max_images must be positive or None.')
    root, output = Path(dataset_dir), Path(output_dir)
    paths = sorted((root/'val/images').glob('*.png'))
    if filenames is not None:
        wanted = set(filenames)
        paths = [p for p in paths if p.name in wanted]
        if {p.name for p in paths} != wanted:
            raise FileNotFoundError('Some explicitly selected WSC files are missing.')
    if max_images is not None:
        paths = paths[:max_images]
    if not paths or (max_images is None and filenames is None and len(paths) != 772):
        raise ValueError('Full WSC requires772 validation images; use an explicit limit for a smoke test.')
    model = get_backbone(cfg).float()
    payload = load_checkpoint(model, checkpoint)
    model.to(device).eval()
    normalizer = SingleImageNormalizer()
    rows = []
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            if rgb.size != (1024, 1024):
                raise ValueError('Native1024 required.')
            normalized = normalizer.normalize(to_tensor(rgb))
        with Image.open(root/'val/semantics'/path.name) as mask:
            gt = remap_semantics(np.array(mask).copy())
        rows.extend({'image': path.name, **r} for r in evaluate_normalized_image(model, normalized, gt, device))
        if (i+1) % 25 == 0 or i+1 == len(paths):
            print(f'[WSC] {i+1}/{len(paths)} images', flush=True)
    image_rows, summary = aggregate_wsc(rows)
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [('wsc_per_offset.csv', rows), ('wsc_per_image.csv', image_rows)]:
        with (output/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    summary.update({'status': 'completed', 'metrics_graph': 'unfused_eval', 'checkpoint': str(Path(checkpoint).resolve()),
                    'dataset_dir': str(root.resolve()),
                    'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    'epoch_0based': payload.get('epoch'), 'source_config': cfg, 'torch': torch.__version__,
                    'device': str(device), 'gpu_name': torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == 'cuda' else None,
                    'full_validation_complete': len(paths) == 772,
                    'normalization': 'actual SingleImageNormalizer once on full1024, then crop1008',
                    'origin': list(ORIGIN), 'crop_size': CROP_SIZE, 'offsets': OFFSETS,
                    'aggregation': 'mean valid offsets per image, then equal-image mean; exclude images with zero valid offsets',
                    'interpretation': 'crop-origin shift stability on GT Weed; includes crop-boundary/context effects; not proof of reduced aliasing'})
    (output/'wsc_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ['config', 'checkpoint', 'dataset-dir', 'output-dir']:
        parser.add_argument('--'+arg, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args()
    run_analysis(args.config, args.checkpoint, args.dataset_dir, args.output_dir, args.device, args.max_images)


if __name__ == '__main__':
    main()
