"""Runtime guards and artifacts for the fresh En/De experiments."""
import csv
import faulthandler
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess

import torch
from pytorch_lightning.callbacks import Callback
import yaml

P2_MODELS = ('NotebookRepDWNetExp1', 'NotebookRepDWNetExp2')
BASELINE_REVISION = 'b53d8ba1ddfd41ebd37c9021b1cb6cdee332084e'
BLOCK_FILES = ('models/rpdnet/RepDWNet.py', 'models/rpdnet/RPD_Module.py',
               'models/rpdnet/rep_dw.py', 'models/rpdnet/RPDNet.py', 'models/rpdnet/RPD_ops.py')


def is_ende(cfg):
    return cfg['backbone']['name'] in P2_MODELS


def validate_fresh_config(cfg, args):
    """Run after CLI overrides, before creating model/optimizer or reading checkpoints."""
    if args.get('resume') or args.get('ckpt_path'):
        raise ValueError('Fresh En/De forbids --resume and --ckpt_path.')
    train, val = cfg['train'], cfg['val']
    required = dict(devices=2, strategy='ddp', optimizer='Adam', max_epoch=100,
                    batch_size=4, accumulate_grad_batches=1, precision=32,
                    learning_rate=2e-4, weight_decay=2e-4, scheduler='cosine',
                    warmup_epochs=5, loss='xentropy', class_weights=[1.47, 5.06, 10.02],
                    start_from_scratch=True, resume_checkpoint=None, drop_last=True)
    for key, expected in required.items():
        if train.get(key) != expected:
            raise ValueError(f'Fresh En/De requires train.{key}={expected!r}, got {train.get(key)!r}')
    if cfg.get('seed') != 1682409321 or train.get('early_stopping', {}).get('mode') != 'none':
        raise ValueError('Fresh En/De requires the B1 seed and disabled early stopping.')
    if args.get('early_stopping_mode') not in (None, 'none'):
        raise ValueError('CLI must not enable early stopping for fresh En/De.')
    if val.get('batch_size') != 4 or val.get('check_val_every_n_epoch') != 5 or not val.get('drop_last'):
        raise ValueError('Fresh En/De validation requires batch 4, interval 5 and drop_last=True.')
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / 'configs/b1_scenarios/B1_run5_repdwnet.yaml').read_text())
    for section, keys in [('data', ['image_normalizer']),
                          ('train', ['color_data_augmentations', 'geometric_data_augmentations']),
                          ('val', ['geometric_data_augmentations'])]:
        for key in keys:
            if cfg[section][key] != baseline[section][key]:
                raise ValueError(f'{section}.{key} must match B1 Run 5.')
    if cfg['backbone'].get('pretrained') or cfg['backbone'].get('deploy'):
        raise ValueError('Fresh En/De must initialize a training model without pretrained weights.')
    if cfg['tracking']['io_retry']['enabled']:
        raise ValueError('I/O retry is disabled in this runner; never skip or replace a failed sample.')
    if cfg['tracking']['source_revision'] != BASELINE_REVISION:
        raise ValueError('The baseline source revision must stay pinned.')
    hashes = {}
    for relative in BLOCK_FILES:
        original = subprocess.check_output(['git', 'show', f'{BASELINE_REVISION}:{relative}'], cwd=root)
        current = (root / relative).read_bytes()
        if original != current:
            raise ValueError(f'B1 Run 5 block has changed from the source pin: {relative}')
        hashes[relative] = hashlib.sha256(current).hexdigest()
    return hashes


def check_dataset(cfg):
    root = Path(cfg['data']['path_to_dataset'])
    counts = {}
    for split in ('train', 'val'):
        image_dir, mask_dir = root / split / 'images', root / split / 'semantics'
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f'Missing image/semantic directory for {split}: {root}')
        # Match datasets.pdc_common.is_image, including its case sensitivity.
        names = sorted(p.name for p in image_dir.iterdir() if p.name.endswith(('.jpg', '.png')))
        if not names:
            raise ValueError(f'No images in {image_dir}')
        missing = [name for name in names if not (mask_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} semantic masks missing; first: {missing[0]}')
        counts[split] = len(names)
    if counts['val'] != cfg['tracking']['expected_num_val_images']:
        raise ValueError(f'Expected 772 validation images, found {counts["val"]}')
    return counts


class FreshEnDeCallback(Callback):
    def __init__(self, cfg, args, block_hashes):
        self.cfg, self.args, self.block_hashes = cfg, args, block_hashes
        self.manifest = None

    def _write(self, trainer):
        if trainer.is_global_zero and self.manifest is not None:
            path = Path(self.args['export_dir']) / 'experiment_manifest.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.manifest, indent=2, ensure_ascii=False) + '\n')
            temporary.replace(path)

    def on_fit_start(self, trainer, pl_module):
        print(f"[RUNTIME rank={trainer.global_rank}] fit setup: WORLD_SIZE={trainer.world_size}", flush=True)
        if trainer.world_size != 2 or trainer.num_devices != 2 or str(trainer.accelerator.__class__.__name__) != 'CUDAAccelerator':
            raise RuntimeError('Main fresh En/De run requires 2 CUDA GPUs and WORLD_SIZE=2.')
        if int(os.environ.get('WORLD_SIZE', '0')) != 2:
            raise RuntimeError('WORLD_SIZE=2 must be supplied by torchrun.')
        if int(os.environ.get('LOCAL_RANK', '-1')) != trainer.local_rank:
            raise RuntimeError('LOCAL_RANK must be supplied by torchrun.')
        if trainer.current_epoch != 0 or trainer.global_step != 0:
            raise RuntimeError('Fresh En/De must start at epoch/global_step 0.')
        if trainer.accumulate_grad_batches != 1 or str(trainer.precision) != '32':
            raise RuntimeError('Resolved precision/accumulation differs from B1 recipe.')
        if not isinstance(trainer.optimizers[0], torch.optim.Adam) or trainer.optimizers[0].state:
            raise RuntimeError('Fresh En/De requires a new Adam optimizer.')
        counts = check_dataset(self.cfg)
        dm = trainer.datamodule
        if len(dm.train_ds) != counts['train'] or len(dm.val_ds) != counts['val']:
            raise RuntimeError('Runtime dataset counts differ from the manifest.')
        for loader in (dm.train_dataloader(), dm.val_dataloader()):
            if loader.batch_size != 4 or not loader.drop_last:
                raise RuntimeError('Runtime loaders must use batch 4 and drop_last=True.')
        path = Path(self.args['export_dir']) / 'experiment_manifest.json'
        occupied = (path.exists() or (path.parent / 'val_history.csv').exists()
                    or bool(list((path.parent / 'checkpoints').glob('*.ckpt')))) if trainer.is_global_zero else False
        if trainer.strategy.broadcast(occupied):
            raise FileExistsError('Use a new run directory; existing training artifacts must be preserved.')
        if not trainer.is_global_zero:
            return
        root = Path(__file__).resolve().parents[1]
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        processed = (math.ceil(counts['val'] / trainer.world_size) // 4) * 4 * trainer.world_size
        network = pl_module.network
        self.manifest = dict(
            scenario=self.cfg['experiment']['id'], backbone=self.cfg['backbone']['name'],
            run_id=path.parent.name, baseline='B1_run5_repdwnet', source_revision=BASELINE_REVISION,
            runtime_source_revision=revision, block_sha256=self.block_hashes,
            notebook_revision=self.cfg['tracking'].get('notebook_revision'),
            executed_cell_sha256=self.cfg['tracking'].get('executed_cell_sha256'),
            architecture=network.channel_table, num_parameters=sum(p.numel() for p in network.parameters()),
            start_from_scratch=True, resume_checkpoint=None, seed=self.cfg['seed'],
            recipe=self.cfg['train'], resolved=dict(world_size=trainer.world_size, num_gpus=trainer.num_devices,
            local_ranks=[0, 1], batch_per_gpu=4, accumulation=1, effective_batch_size=8,
            precision=trainer.precision, optimizer=type(trainer.optimizers[0]).__name__),
            dataset=dict(path=self.cfg['data']['path_to_dataset'], num_train_images=counts['train'],
            num_val_images=counts['val'], processed_val_images=processed,
            train_drop_last=True, val_drop_last=True), io_retry=self.cfg['tracking']['io_retry'],
            status='running', validation_epochs=[], checkpoints={})
        self._write(trainer)

    def on_sanity_check_start(self, trainer, pl_module):
        print(f'[RUNTIME rank={trainer.global_rank}] sanity validation started', flush=True)

    def on_sanity_check_end(self, trainer, pl_module):
        print(f'[RUNTIME rank={trainer.global_rank}] sanity validation finished', flush=True)

    def on_train_start(self, trainer, pl_module):
        faulthandler.cancel_dump_traceback_later()
        print(f'[RUNTIME rank={trainer.global_rank}] training started: epoch={trainer.current_epoch}', flush=True)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f'[EPOCH] start {trainer.current_epoch}', flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(f'[EPOCH] end {trainer.current_epoch}', flush=True)
        if self.manifest is not None:
            self.manifest['last_completed_train_epoch'] = trainer.current_epoch
            self._write(trainer)

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        history = pl_module.val_history
        epochs = [int(row['epoch']) for row in history]
        self.manifest['validation_epochs'] = epochs
        for callback in trainer.checkpoint_callbacks:
            if callback.best_model_path:
                score = callback.best_model_score
                match = re.search(r'epoch(?:epoch=|=)?(\d+)', Path(callback.best_model_path).name)
                self.manifest['checkpoints'][callback.monitor or 'periodic'] = dict(
                    path=callback.best_model_path, metric=float(score) if score is not None else None,
                    epoch=int(match.group(1)) if match else None)
            if callback.last_model_path:
                self.manifest['checkpoints']['last'] = dict(path=callback.last_model_path,
                    epoch=self.manifest.get('last_completed_train_epoch'))
        expected = self.cfg['tracking']['validation_history']['expected_epochs']
        complete = epochs == expected and self.manifest.get('last_completed_train_epoch') == 99
        self.manifest['status'] = 'complete' if complete else 'failed'
        if not complete:
            self.manifest['failure_reason'] = f'Incomplete fresh run: expected epochs {expected}, got {epochs}'
        self._write(trainer)
        if not complete:
            raise RuntimeError(self.manifest['failure_reason'])

    def on_exception(self, trainer, pl_module, exception):
        if trainer.is_global_zero:
            if self.manifest is None:
                self.manifest = dict(scenario=self.cfg['experiment']['id'], start_from_scratch=True,
                                     resume_checkpoint=None)
                # Never overwrite an old manifest when directory reuse was rejected.
                if (Path(self.args['export_dir']) / 'experiment_manifest.json').exists():
                    return
            self.manifest.update(status='failed', failure_reason=str(exception))
            self._write(trainer)
