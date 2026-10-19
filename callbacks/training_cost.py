"""Observed fit cost only; no model, optimizer, scheduler or sampling changes."""
import json
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback


class TrainingCostCallback(Callback):
    def __init__(self, output_dir, experiment_id):
        self.output = Path(output_dir)
        self.experiment_id = experiment_id

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _memory(self):
        if self.device.type == 'cuda':
            self.allocated = max(self.allocated, torch.cuda.max_memory_allocated(self.device))
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved(self.device))

    def on_fit_start(self, trainer, pl_module):
        # Retain the training device: teardown can move the module to CPU.
        self.device = pl_module.device
        self._sync()
        self.started = time.perf_counter()
        self.start_epoch = trainer.current_epoch
        self.epochs = []
        self.allocated = self.reserved = 0
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.epochs:
            self.start_epoch = trainer.current_epoch
            self.start_global_step = trainer.global_step
        self._sync()
        self.epoch_started = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._memory()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._memory()

    def on_train_epoch_end(self, trainer, pl_module):
        self._sync()
        self._memory()
        self.epochs.append({'epoch_0based': trainer.current_epoch,
                            'seconds': time.perf_counter()-self.epoch_started})

    def on_fit_end(self, trainer, pl_module):
        self._sync()
        self._memory()
        gpu = self.device.type == 'cuda'
        seconds = time.perf_counter()-self.started
        record = {'experiment': self.experiment_id, 'status': 'completed',
                  'timing_scope': 'this fit invocation, per rank; not cumulative across resumes',
                  'epoch_timing_scope': 'on_train_epoch_start to on_train_epoch_end, including intra-epoch validation',
                  'rank': trainer.global_rank, 'world_size': trainer.world_size,
                  'device': str(self.device),
                  'gpu_name': torch.cuda.get_device_name(self.device) if gpu else None,
                  'start_epoch_0based': self.start_epoch,
                  'start_global_step': getattr(self, 'start_global_step', None),
                  'total_fit_seconds': seconds, 'epochs': self.epochs,
                  'mean_epoch_wall_seconds': sum(x['seconds'] for x in self.epochs)/len(self.epochs) if self.epochs else None,
                  'peak_gpu_allocated_bytes': self.allocated if gpu else None,
                  'peak_gpu_reserved_bytes': self.reserved if gpu else None,
                  'trainable_parameters': sum(p.numel() for p in pl_module.network.parameters() if p.requires_grad),
                  'persistent_buffer_elements': sum(v.numel() for k,v in pl_module.network.state_dict().items()
                                                   if k not in dict(pl_module.network.named_parameters()))}
        ranks = [record]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ranks = [None]*trainer.world_size
            torch.distributed.all_gather_object(ranks, record)
        directory = self.output/'training_cost'
        directory.mkdir(parents=True, exist_ok=True)
        invocation = f"start{self.start_epoch:03d}_step{trainer.global_step:08d}"
        (directory/f'{invocation}_rank{trainer.global_rank}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        if trainer.is_global_zero:
            summary = {**record, 'ranks': ranks,
                       'peak_gpu_allocated_bytes_rank_zero': record['peak_gpu_allocated_bytes'],
                       'peak_gpu_reserved_bytes_rank_zero': record['peak_gpu_reserved_bytes']}
            text = json.dumps(summary, indent=2, allow_nan=False)
            (directory/f'{invocation}.json').write_text(text)
            (self.output/'training_cost.json').write_text(text)
