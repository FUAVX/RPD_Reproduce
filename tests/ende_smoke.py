"""CPU DDP smoke test; synthetic data and output are never ablation results.

Run with torchrun --standalone --nproc_per_node=2 tests/ende_smoke.py
  --config configs/p2_scenarios/P2_EncoderHalf.yaml --output /tmp/ende-smoke-en
"""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models import get_backbone, get_criterion, model_multimetrics
from callbacks.config_callback import ConfigCallback
from utils.ende_training import validate_fresh_config


class SyntheticImages(Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return dict(input_image=torch.randn(3, 32, 32, generator=generator),
                    anno=torch.randint(0, 3, (32, 32), generator=generator))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    validate_fresh_config(cfg, {})
    for forbidden in ({'resume': True}, {'ckpt_path': 'old.ckpt'}):
        try:
            validate_fresh_config(cfg, forbidden)
        except ValueError:
            pass
        else:
            raise AssertionError('Fresh checkpoint guard failed')
    torch.set_num_threads(1)
    seed_everything(cfg['seed'])
    network = get_backbone(cfg)
    module = model_multimetrics.SegmentationNetwork(
        network, get_criterion(cfg), 2e-4, 2e-4,
        train_step_settings=['regular'], val_step_settings=['regular'],
        scheduler_type='cosine', warmup_epochs=5)
    callbacks = [ModelCheckpoint(dirpath=Path(args.output)/'checkpoints', monitor=monitor,
                 mode='max', filename=monitor+'_{epoch:02d}', save_top_k=1)
                 for monitor in ('val_mIoU', 'val_weed_iou')]
    callbacks.append(ConfigCallback(cfg))
    callbacks.append(ModelCheckpoint(dirpath=Path(args.output)/'checkpoints', save_top_k=0,
                                    save_last=True, save_on_train_epoch_end=True))
    trainer = Trainer(accelerator='cpu', devices=2, strategy='ddp', max_epochs=10,
        check_val_every_n_epoch=5, num_sanity_val_steps=2,
        limit_train_batches=1, limit_val_batches=1, enable_progress_bar=False,
        logger=TensorBoardLogger(args.output, name='lightning_logs', version=''),
        default_root_dir=args.output, callbacks=callbacks)
    loader = DataLoader(SyntheticImages(), batch_size=4, drop_last=True)
    trainer.fit(module, train_dataloaders=loader, val_dataloaders=loader)
    if trainer.is_global_zero:
        with (Path(args.output)/'val_history.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in rows] == [4, 9]
        for callback in trainer.checkpoint_callbacks:
            path = callback.best_model_path or callback.last_model_path
            assert Path(path).is_file()
        print('PASS: 2 CPU ranks, En/De training + sanity exclusion + history [4, 9] + checkpoints')
    else:
        assert module.val_history == []


if __name__ == '__main__':
    main()
