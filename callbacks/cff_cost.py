"""Observed campaign cost only; no synthetic steps, RNG or optimizer mutation."""
from pathlib import Path
import json
import time
import torch
from pytorch_lightning import Callback

class CFFCostCallback(Callback):
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def on_fit_start(self, trainer, pl_module):
        self.started = time.perf_counter()
        self.device = pl_module.device
        self.epochs = []
        self.start_epoch = None
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.start_epoch is None:
            self.start_epoch = trainer.current_epoch
            self.start_step = trainer.global_step
        self.epoch_started = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self.epochs.append({'epoch_0based': trainer.current_epoch,
                            'seconds': time.perf_counter() - self.epoch_started})

    def on_fit_end(self, trainer, pl_module):
        seconds = time.perf_counter() - self.started
        network = pl_module.network
        aux = 0  # CFF has a single head and no auxiliary objective
        total = sum(p.numel() for p in network.parameters() if p.requires_grad)
        record = {'status': 'completed', 'timing_scope': 'this fit invocation per rank, including validation; not cumulative across resumes',
                  'total_fit_seconds': seconds, 'rank': trainer.global_rank, 'world_size': trainer.world_size,
                  'start_epoch_0based': self.start_epoch, 'start_global_step': getattr(self, 'start_step', None),
                  'end_global_step': trainer.global_step, 'epochs': self.epochs,
                  'trainable_params': total, 'main_trainable_params': total-aux, 'aux_trainable_params': aux,
                  'device': str(self.device), 'gpu_name': torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else None,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(self.device) if self.device.type=='cuda' else None}
        target = self.output_dir/'training_cost'; target.mkdir(parents=True, exist_ok=True)
        (target/f'rank{trainer.global_rank}_start{self.start_epoch}_endstep{trainer.global_step}.json').write_text(json.dumps(record, indent=2)+'\n')
        if trainer.is_global_zero:
            (self.output_dir/'training_cost.json').write_text(json.dumps(record, indent=2)+'\n')
