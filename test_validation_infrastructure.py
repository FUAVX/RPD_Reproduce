"""D02/D03 acceptance: unequal rank losses, real sanity checks, writers and resume.

python test_validation_infrastructure.py
python test_validation_infrastructure.py --ddp --output-dir DIR --accelerator cpu --devices 2
python test_validation_infrastructure.py --ddp --output-dir DIR --accelerator gpu --devices 2 --resume
"""
import argparse
from copy import deepcopy
import csv
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import models.model_multimetrics as shared
from utils.validation_artifacts import VALIDATION_FIELDS, normalize_history


class ProbeNetwork(nn.Module):
    num_classes = 3

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x):
        return self.conv(x)


class ProbeCriterion(nn.Module):
    def forward(self, logits, labels, mode, **kwargs):
        return nn.functional.cross_entropy(logits, labels)


class ProbeData(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return {'input_image': torch.randn(3, 8, 8, generator=generator),
                'anno': torch.randint(0, 3, (8, 8), generator=generator)}


class ValidationProbe(shared.SegmentationNetwork):
    def __init__(self):
        super().__init__(ProbeNetwork(), ProbeCriterion(), .0002, .0002,
                         train_step_settings=['regular'], val_step_settings=['regular'],
                         test_step_settings=['regular'])
        self.sanity_batches = 0
        self.restored_history = None

    def validation_step(self, batch, batch_idx):
        result = super().validation_step(batch, batch_idx)
        if self.trainer.sanity_checking:
            self.sanity_batches += 1
        # Deliberately unequal across ranks: global rank mean = 0.3 + epoch*0.01.
        result['loss'] = torch.tensor(.2 * (self.global_rank + 1) + .01 * self.current_epoch,
                                      device=self.device)
        return result

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        if self.val_history and self.trainer.world_size > 1:
            values = torch.tensor([[r[key] for key in VALIDATION_FIELDS]
                                   for r in self.val_history], device=self.device)
            gathered = self.all_gather(values)
            assert torch.equal(gathered[0], gathered[1])

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        assert self.val_history == checkpoint['val_history']
        self.restored_history = deepcopy(self.val_history)


class SanityAudit(Callback):
    def on_sanity_check_start(self, trainer, pl_module):
        self.before = deepcopy(pl_module.val_history)

    def on_sanity_check_end(self, trainer, pl_module):
        assert pl_module.sanity_batches > 0
        assert self.before == pl_module.val_history
        assert not pl_module.validation_step_outputs


def ddp_acceptance(output, accelerator='cpu', devices=2, resume=False):
    pl.seed_everything(1682409321)
    output = Path(output)
    net = ValidationProbe()
    logger = TensorBoardLogger(str(output), name='lightning_logs', version=0,
                               default_hp_metric=False)
    checkpoint = ModelCheckpoint(dirpath=str(output / 'checkpoints'), save_last=True,
                                 monitor='val_loss', mode='min', save_top_k=1)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=90)),
        max_epochs=3 if resume else 2, num_sanity_val_steps=2,
        logger=logger, callbacks=[checkpoint, SanityAudit()],
        default_root_dir=str(output), enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1, limit_train_batches=2)
    writes = []
    original_csv, original_yaml = shared.write_validation_history, shared.save_metric

    def write_csv(*args, **kwargs):
        assert trainer.is_global_zero, 'Nonzero rank attempted a shared CSV write'
        writes.append('csv')
        return original_csv(*args, **kwargs)

    def write_yaml(*args, **kwargs):
        assert trainer.is_global_zero, 'Nonzero rank attempted a shared YAML write'
        writes.append('yaml')
        return original_yaml(*args, **kwargs)

    loader = DataLoader(ProbeData(), batch_size=2)
    with patch.object(shared, 'write_validation_history', side_effect=write_csv), \
         patch.object(shared, 'save_metric', side_effect=write_yaml):
        trainer.fit(net, loader, loader,
                    ckpt_path=str(output / 'checkpoints/last.ckpt') if resume else None)
        # Covers the test/evaluation YAML writer as well as validation.
        monitored_loss = float(trainer.callback_metrics['val_loss'])
        trainer.test(net, dataloaders=loader, verbose=False)
    expected_epochs = [0, 1, 2] if resume else [0, 1]
    assert [r['epoch'] for r in net.val_history] == expected_epochs
    if resume:
        assert [r['epoch'] for r in net.restored_history] == [0, 1]
    for row in net.val_history:
        assert abs(row['val_loss'] - (.3 + .01 * row['epoch'])) < 5.1e-6
    if trainer.is_global_zero:
        with (output / 'val_history.csv').open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert [int(r['epoch']) for r in csv_rows] == expected_epochs
        state = torch.load(output / 'checkpoints/last.ckpt', map_location='cpu', weights_only=True)
        assert state['val_history'] == net.val_history
        assert abs(monitored_loss - net.val_history[-1]['val_loss']) < 5.1e-6
        monitor_state = next(v for v in state['callbacks'].values()
                             if isinstance(v, dict) and v.get('monitor') == 'val_loss')
        assert abs(float(monitor_state['best_model_score']) - .3) < 5.1e-6
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(logger.log_dir, size_guidance={'scalars': 0}).Reload()
        epoch_by_step = {v.step: int(v.value) for v in events.Scalars('epoch')}
        for value in events.Scalars('val_loss'):
            assert abs(value.value - (.3 + .01 * epoch_by_step[value.step])) < 5.1e-6
        assert len(events.Scalars('val_loss')) == len(expected_epochs)
        assert writes.count('csv') == (1 if resume else 2)
        assert writes.count('yaml') == (10 if resume else 15)
    else:
        assert writes == []
    # Rank-specific evidence files intentionally have different names.
    report = {'rank': trainer.global_rank, 'resume': resume, 'writers': writes,
              'history': net.val_history, 'sanity_batches': net.sanity_batches,
              'restored_history': net.restored_history, 'status': 'passed'}
    (output / f'acceptance_rank{trainer.global_rank}_resume{int(resume)}.json').write_text(
        json.dumps(report, indent=2))
    print(f'PASS D02/D03 rank={trainer.global_rank} resume={resume}', flush=True)


class InfrastructureTests(unittest.TestCase):
    def row(self, epoch, value=.3):
        return {key: epoch if key == 'epoch' else value for key in VALIDATION_FIELDS}

    def test_history_deduplicates_and_sorts_without_filtering_real_epoch_zero(self):
        rows = normalize_history([self.row(4), self.row(0), self.row(4, .4)])
        self.assertEqual([r['epoch'] for r in rows], [0, 4])
        self.assertEqual(rows[-1]['val_loss'], .4)

    def test_history_rejects_corruption_and_legacy_resume(self):
        for row in [{'epoch': 0}, self.row(-1), self.row(.5), self.row(1, float('nan'))]:
            with self.assertRaises(ValueError):
                normalize_history([row])
        with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
            ValidationProbe().on_load_checkpoint({'epoch': 99, 'global_step': 17600})

    def test_checkpoint_restores_history_and_ignores_no_csv(self):
        net = ValidationProbe(); net.val_history = [self.row(4), self.row(9)]
        checkpoint = {}; net.on_save_checkpoint(checkpoint)
        restored = ValidationProbe(); restored.on_load_checkpoint(checkpoint)
        self.assertEqual(restored.val_history, net.val_history)
        checkpoint['val_history'][0]['val_loss'] = 99
        self.assertNotEqual(restored.val_history[0]['val_loss'], 99)

    def test_incomplete_validation_has_no_persistent_row_or_monitor(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            net = ValidationProbe()
            net._trainer = SimpleNamespace(strategy=SimpleNamespace(reduce=lambda x, **k: x),
                log_dir=directory, default_root_dir=directory, num_val_batches=[2],
                world_size=1, sanity_checking=False, current_epoch=0, global_rank=0, is_global_zero=True)
            net.log = lambda *a, **k: self.fail('Incomplete pass updated checkpoint monitor')
            net.validation_step({'input_image': torch.randn(2, 3, 8, 8),
                                 'anno': torch.zeros(2, 8, 8, dtype=torch.long)}, 0)
            net.on_validation_epoch_end()
            self.assertEqual(net.val_history, [])
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(net.validation_step_outputs, [])


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ddp', action='store_true'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output-dir'); parser.add_argument('--accelerator', default='cpu')
    parser.add_argument('--devices', type=int, default=2)
    args, rest = parser.parse_known_args()
    if args.ddp:
        if not args.output_dir or args.devices != 2:
            parser.error('DDP acceptance requires output-dir and exactly 2 devices')
        ddp_acceptance(args.output_dir, args.accelerator, args.devices, args.resume)
    else:
        unittest.main(argv=[__file__, *rest])
