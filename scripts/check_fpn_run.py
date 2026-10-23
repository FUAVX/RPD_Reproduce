"""Validate a completed campaign's checkpoint/history artifacts before packaging."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from utils.validation_artifacts import normalize_history


def audit_run(output_dir, expected_epochs=100, val_interval=5):
    root = Path(output_dir)
    state = torch.load(root / 'checkpoints/last.ckpt', map_location='cpu', weights_only=True)
    if state['epoch'] != expected_epochs - 1:
        raise ValueError('Last checkpoint has not completed the requested campaign')
    history = normalize_history(state['val_history'])
    expected = list(range(val_interval - 1, expected_epochs, val_interval))
    if [r['epoch'] for r in history] != expected:
        raise ValueError('Checkpoint history missing/duplicating completed validation epochs')
    with (root / 'val_history.csv').open() as stream:
        rows = [{k: int(v) if k == 'epoch' else float(v) for k, v in row.items()}
                for row in csv.DictReader(stream)]
    if rows != history:
        raise ValueError('CSV does not match authoritative checkpoint history')
    policies = []
    for value in state['callbacks'].values():
        if not isinstance(value, dict) or value.get('monitor') not in ('val_loss', 'val_mIoU', 'val_weed_iou'):
            continue
        monitor = value['monitor']; path = root / 'checkpoints' / Path(value['best_model_path']).name
        if not path.is_file():
            raise FileNotFoundError(path)
        selected = torch.load(path, map_location='cpu', weights_only=True)
        epoch = selected['epoch']; field = {'val_loss': 'val_loss', 'val_mIoU': 'mIoU',
                                          'val_weed_iou': 'weed_iou'}[monitor]
        row = next(r for r in history if r['epoch'] == epoch)
        score = float(value['best_model_score'])
        optimum = (min if monitor == 'val_loss' else max)(r[field] for r in history)
        if abs(row[field] - optimum) > 1e-12:
            raise ValueError('Saved checkpoint is not the history optimum')
        if abs(row[field] - score) > 5.1e-6:
            raise ValueError(f'Checkpoint {monitor} differs from synchronized history')
        if selected.get('fusion') != state.get('fusion'):
            raise ValueError('Selected checkpoint Fuse settings differ from campaign')
        if normalize_history(selected['val_history']) != [r for r in history if r['epoch'] <= epoch]:
            raise ValueError('Selected checkpoint contains inconsistent validation history')
        policies.append({'monitor': monitor, 'epoch': epoch, 'score': score, 'checkpoint': path.name})
    if {p['monitor'] for p in policies} != {'val_loss', 'val_mIoU', 'val_weed_iou'}:
        raise ValueError('Missing campaign checkpoint selection policy')
    return {'status': 'passed', 'last_epoch': state['epoch'], 'global_step': state['global_step'],
            'validation_epochs': expected, 'history_source': 'checkpoint',
            'csv_checkpoint_parity': True, 'checkpoint_selections': policies,
            'note': 'Artifact audit, not an independent dataset metric reevaluation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True); parser.add_argument('--report', required=True)
    parser.add_argument('--expected-epochs', type=int, default=100)
    parser.add_argument('--val-interval', type=int, default=5)
    args = parser.parse_args()
    report = audit_run(args.output_dir, args.expected_epochs, args.val_interval)
    path = Path(args.report); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
