"""PSPNet-inspired training-only head; team C/2 adaptation, fixed Now protocol."""
from copy import deepcopy
from pathlib import Path
import csv
import torch
from torch import nn

REFERENCE_COMMIT = '4f274c3f276778228bc14a4565822d46359f0cc8'
TAP_CHANNELS = {'x3': 64, 'x4': 128, 'x5': 128, 'x6': 64, 'x7': 32}

def validate_deep_supervision(config):
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError('deep_supervision must be a mapping or None')
    fixed = {'weight': 0.4, 'hidden_ratio': 0.5, 'dropout': 0.1, 'align_corners': True}
    if set(config) != {'tap', *fixed} or config.get('tap') not in TAP_CHANNELS:
        raise ValueError('DeepSup-Now requires one tap x3..x7 and explicit fixed settings')
    if any(type(config[k]) is not type(v) or config[k] != v for k, v in fixed.items()):
        raise ValueError(f'DeepSup-Now fixed settings: {fixed}')
    return deepcopy(config)

class AuxiliaryHead(nn.Sequential):
    """Official Conv3/BN/ReLU/Dropout/Conv1 topology, with team hidden=C/2."""
    def __init__(self, channels, classes=3):
        if channels not in (32, 64, 128) or classes != 3:
            raise ValueError('DeepSup-Now is locked to RepDWNet tap widths and 3 classes')
        hidden = channels // 2
        super().__init__(nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
                         nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
                         nn.Dropout2d(0.1), nn.Conv2d(hidden, classes, 1, bias=True))

def record_objectives(module, outputs):
    """Detached epoch means, one DDP reduction, rank-zero CSV/TensorBoard writes."""
    names = ('train_main_loss', 'train_aux_loss', 'train_total_loss')
    values = torch.stack([torch.stack([o[n] for n in names]) for o in outputs]).mean(0)
    if module.trainer.world_size > 1:
        values = module.all_gather(values).reshape(-1, 3).mean(0)
    for name, value in zip(names, values):
        module.log(name, value, on_step=False, on_epoch=True, sync_dist=False)
    if module.trainer.is_global_zero:
        path = Path(module.trainer.default_root_dir) / 'deep_supervision_history.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', *names])
            if new: writer.writeheader()
            writer.writerow({'epoch': module.current_epoch,
                             **{n: float(v.cpu()) for n, v in zip(names, values)}})
