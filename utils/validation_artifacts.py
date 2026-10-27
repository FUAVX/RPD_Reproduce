"""Shared D02/D03 artifact serialization; checkpoint history is authoritative."""
import csv
import math
import os
from pathlib import Path
import tempfile

import yaml

VALIDATION_FIELDS = (
    'epoch', 'train_loss', 'val_loss', 'mIoU', 'soil_iou', 'crop_iou',
    'weed_iou', 'weed_precision', 'weed_recall', 'weed_f1', 'lr',
)


def normalize_history(rows):
    """Validate complete rows, keep the latest value per epoch, sort deterministically."""
    if not isinstance(rows, list):
        raise ValueError('Checkpoint val_history must be a list of completed validation rows')
    by_epoch = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(VALIDATION_FIELDS):
            raise ValueError('Incomplete or unexpected validation history fields')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in row.values()):
            raise ValueError('Validation history must contain finite numeric values')
        epoch = row['epoch']
        if epoch < 0 or int(epoch) != epoch:
            raise ValueError('Validation history epoch must be a nonnegative integer')
        normalized = {key: row[key] for key in VALIDATION_FIELDS}
        normalized['epoch'] = int(epoch)
        by_epoch[int(epoch)] = normalized
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _atomic_write(path, serialize):
    """Publish one complete file; readers never observe a partially rewritten CSV/YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = stream.name
            serialize(stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def write_validation_history(path, rows):
    rows = normalize_history(rows)

    def serialize(stream):
        writer = csv.DictWriter(stream, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, serialize)


def write_metric(path, metrics, name):
    info = {f'class_{i}': round(float(value), 5) for i, value in enumerate(metrics)}
    info[f'm{name}'] = round(float(metrics.mean()), 5)
    _atomic_write(path, lambda stream: yaml.safe_dump(info, stream))
