"""DPP parameter histories and measured training cost, with rank-zero exports."""

import json
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback

from models.rpdnet.dpp import LiteDPPSym


def parameter_record(alpha, exponent):
    record = {}
    for key, tensor in (('alpha', alpha), ('lambda', exponent)):
        values = tensor.detach().float().cpu().flatten()
        if not torch.isfinite(values).all():
            raise ValueError(f'Non-finite DPP {key} values.')
        record[key] = {
            'values': values.tolist(), 'mean': values.mean().item(),
            'median': values.quantile(0.5).item(), 'std': values.std(unbiased=False).item(),
            'min': values.min().item(), 'max': values.max().item(),
        }
    return record


class DPPLoggingCallback(Callback):
    def __init__(self, output_dir, experiment_id):
        self.output_dir = Path(output_dir) / 'dpp_parameters'
        self.experiment_id = experiment_id

    def _snapshot(self, trainer, pl_module, event):
        if not trainer.is_global_zero:
            return
        layers = {name: parameter_record(module.alpha, module.exponent)
                  for name, module in pl_module.network.named_modules()
                  if isinstance(module, LiteDPPSym)}
        record = {'experiment': self.experiment_id, 'event': event,
                  'epoch': trainer.current_epoch, 'global_step': trainer.global_step, 'layers': layers}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / 'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        (self.output_dir / f'{event}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
        if trainer.logger and event == 'epoch_end':
            scalars = {f'dpp/{name}/{kind}/{stat}': value
                       for name, layer in layers.items() for kind, stats in layer.items()
                       for stat, value in stats.items() if stat != 'values'}
            trainer.logger.log_metrics(scalars, step=trainer.global_step)

    @staticmethod
    def _synchronize(pl_module):
        if pl_module.device.type == 'cuda':
            torch.cuda.synchronize(pl_module.device)

    def on_fit_start(self, trainer, pl_module):
        self._synchronize(pl_module)
        self.fit_started = time.perf_counter()
        self.epoch_times = []
        if pl_module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self._snapshot(trainer, pl_module, 'initial')

    def on_train_epoch_start(self, trainer, pl_module):
        self._synchronize(pl_module)
        self.epoch_started = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        self._synchronize(pl_module)
        self.epoch_times.append(time.perf_counter() - self.epoch_started)
        self._snapshot(trainer, pl_module, 'epoch_end')

    def on_fit_end(self, trainer, pl_module):
        self._synchronize(pl_module)
        if not trainer.is_global_zero:
            return
        self._snapshot(trainer, pl_module, 'final')
        gpu = pl_module.device.type == 'cuda'
        cost = {
            'experiment': self.experiment_id, 'timing_scope': 'this fit invocation, rank zero',
            'total_fit_seconds': time.perf_counter() - self.fit_started,
            'epoch_wall_seconds_including_validation': self.epoch_times,
            'mean_epoch_wall_seconds': sum(self.epoch_times) / max(len(self.epoch_times), 1),
            'peak_gpu_allocated_bytes_rank_zero': torch.cuda.max_memory_allocated(pl_module.device) if gpu else None,
            'peak_gpu_reserved_bytes_rank_zero': torch.cuda.max_memory_reserved(pl_module.device) if gpu else None,
            'world_size': trainer.world_size,
            'trainable_parameters': sum(p.numel() for p in pl_module.network.parameters() if p.requires_grad),
        }
        (self.output_dir / 'training_cost.json').write_text(json.dumps(cost, indent=2))
        self._export_checkpoints(trainer)

    def _export_checkpoints(self, trainer):
        paths = {}
        for callback in trainer.checkpoint_callbacks:
            path = getattr(callback, 'best_model_path', '')
            monitor = getattr(callback, 'monitor', None)
            if path and monitor in ('val_weed_iou', 'val_mIoU', 'val_loss'):
                paths[monitor] = path
            last_path = getattr(callback, 'last_model_path', '')
            if last_path:
                paths['last'] = last_path
        for label, path in paths.items():
            # Load only our training artifacts; weights_only=False supports Lightning metadata.
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            state = checkpoint['state_dict']
            layers = {}
            for key, value in state.items():
                if key.startswith('network.') and key.endswith('.log_alpha'):
                    prefix = key[:-len('.log_alpha')]
                    layers[prefix[len('network.'):]] = parameter_record(
                        value.clamp(max=10).exp(), state[prefix + '.log_lambda'].exp())
            record = {'experiment': self.experiment_id, 'checkpoint': path,
                      'epoch': checkpoint.get('epoch'), 'global_step': checkpoint.get('global_step'),
                      'layers': layers}
            (self.output_dir / f'checkpoint_{label}.json').write_text(json.dumps(record, indent=2, allow_nan=False))
