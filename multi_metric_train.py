"""Train semantic segmentation model."""

import argparse
import sys
from calendar import c
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import warnings

# Tắt tất cả warnings rác từ thư viện cũ
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# NumPy 2.0 compatibility patch for older PyTorch Lightning / TorchMetrics
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _alias, _target in [('Inf', np.inf), ('Infinity', np.inf), ('infty', np.inf),
                            ('NaN', np.nan), ('bool', bool), ('int', int), ('float', float)]:
        try:
            if not hasattr(np, _alias):
                setattr(np, _alias, _target)
        except Exception:
            pass

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, Callback

from callbacks import (ConfigCallback, PostprocessorCallback, VisualizerCallback, get_postprocessors, get_visualizers, ComputeProfilingCallback)
from datasets import get_data_module
from models import get_backbone, get_criterion, model_multimetrics

import torch


class PaperEarlyStopping(Callback):
    """
    Early stopping mechanism exactly as defined in the IEEE TGRS 2026 Paper:
    "an early stopping mechanism after three consecutive validations with a loss <= 0.1, validating every 200 epochs."
    """
    def __init__(self, loss_threshold: float = 0.1, consecutive_count: int = 3, monitor: str = 'val_loss', verbose: bool = True):
        super().__init__()
        self.loss_threshold = loss_threshold
        self.consecutive_count = consecutive_count
        self.monitor = monitor
        self.verbose = verbose
        self.current_consecutive = 0

    def on_validation_end(self, trainer: "Trainer", pl_module: "model_multimetrics.SegmentationNetwork") -> None:
        if getattr(trainer, 'sanity_checking', False):
            return

        logs = trainer.callback_metrics
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return

        current_loss_val = float(current_loss.item() if hasattr(current_loss, 'item') else current_loss)
        if current_loss_val <= self.loss_threshold:
            self.current_consecutive += 1
            if self.verbose:
                print(f"\n[Paper EarlyStopping] {self.monitor} = {current_loss_val:.4f} <= {self.loss_threshold} "
                      f"({self.current_consecutive}/{self.consecutive_count} consecutive validations)")
            if self.current_consecutive >= self.consecutive_count:
                if self.verbose:
                    print(f"\n[Paper EarlyStopping] TRIGGERED! {self.consecutive_count} consecutive validations "
                          f"with {self.monitor} <= {self.loss_threshold}. Stopping training at epoch {trainer.current_epoch}.")
                trainer.should_stop = True
        else:
            if self.current_consecutive > 0 and self.verbose:
                print(f"\n[Paper EarlyStopping] {self.monitor} = {current_loss_val:.4f} > {self.loss_threshold}. "
                      f"Resetting consecutive counter from {self.current_consecutive} to 0.")
            self.current_consecutive = 0

    def state_dict(self) -> Dict[str, Any]:
        return {'current_consecutive': self.current_consecutive}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_consecutive = state_dict.get('current_consecutive', 0)


class TeeStream:
    """Duplicate stdout/stderr to a log file while keeping terminal output active."""
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        try:
            self.log_file.write(data)
            self.log_file.flush()
        except Exception:
            pass

    def flush(self):
        self.original_stream.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self.original_stream, 'isatty', lambda: False)()

    def fileno(self):
        return self.original_stream.fileno()


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description='Train RPD Semantic Segmentation Model')
    parser.add_argument('--export_dir', default='log_dir/', help='Path to export dir which saves logs, metrics, etc.')
    parser.add_argument('--config', default='./config/config_deeplearn.yaml',
                        help="Path to configuration file (*.yaml)")
    parser.add_argument('--ckpt_path', default=None, help='Provide *.ckpt file to continue training.')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training state from checkpoint')
    
    # Các tham số tiện lợi khi chạy trên server SSH
    parser.add_argument('--dataset_dir', default=None, type=str,
                        help='Override path_to_dataset in config (e.g. /data/PhenoBench)')
    parser.add_argument('--batch_size', default=None, type=int,
                        help='Override training batch_size (default: from config, e.g. 4)')
    parser.add_argument('--max_epoch', default=None, type=int,
                        help='Override training max_epoch (default: from config)')
    parser.add_argument('--devices', default=None, type=str,
                        help='Override GPU devices (e.g. 1, "0,", "auto")')
    parser.add_argument('--num_workers', default=None, type=int,
                        help='Override DataLoader num_workers (e.g. 8 for high-RAM server)')
    parser.add_argument('--learning_rate', default=None, type=float,
                        help='Override learning rate (default: from config, paper is 1e-4)')
    parser.add_argument('--check_val_every_n_epoch', default=None, type=int,
                        help='Override validation frequency (paper protocol is 200)')
    parser.add_argument('--early_stopping_mode', default=None, choices=['paper', 'plateau', 'none'],
                        help="Early stopping mode: 'paper' (loss <= 0.1 for 3 consecutive checks), 'plateau' (patience without improvement), 'none'")
    parser.add_argument('--early_stopping_threshold', default=None, type=float,
                        help='Loss threshold for paper early stopping (default: 0.1)')
    parser.add_argument('--early_stopping_consecutive', default=None, type=int,
                        help='Consecutive validation count for paper early stopping (default: 3)')
    parser.add_argument('--early_stopping_patience', default=None, type=int,
                        help='Patience for plateau EarlyStopping (set 0 or negative to disable EarlyStopping)')
    parser.add_argument('--no_early_stopping', default=False, action='store_true',
                        help='Completely disable EarlyStopping to train for full epochs')
    parser.add_argument('--precision', default=None, type=str,
                        help='Override training precision: 32, 16, or 16-mixed')
    parser.add_argument('--scheduler', default=None, choices=['poly', 'cosine'],
                        help="Override LR scheduler: 'poly' (polynomial decay) or 'cosine' (cosine annealing)")
    parser.add_argument('--warmup_epochs', default=None, type=int,
                        help='Override warmup epochs count (default: from config, e.g. 5)')
    parser.add_argument('--accumulate_grad_batches', default=None, type=int,
                        help='Override gradient accumulation steps (default: 1)')
    parser.add_argument('--profile', default=False, action='store_true',
                        help='Enable automated B0 compute profiling callback (measures VRAM, steady speed, overhead, and runtime estimates)')
    parser.add_argument('--profile_warmup_batches', default=10, type=int,
                        help='Number of initial batches to exclude as warm-up from steady-state profiling (default: 10)')

    args = vars(parser.parse_args())
    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    # Tự động dò tìm dataset PhenoBench trên Kaggle nếu đường dẫn trong YAML không tồn tại
    current_path = config.get('data', {}).get('path_to_dataset', '')
    if not os.path.exists(current_path):
        candidates = [
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset/PhenoBench",
            "/kaggle/input/phenobench-dataset/PhenoBench",
            "/kaggle/input/datasets/ndhoang2310/phenobench-dataset",
            "/kaggle/input/phenobench-dataset",
        ]
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "train", "images")):
                print(f"[Auto-Detect] Thay thế đường dẫn dataset thành: {cand}")
                config['data']['path_to_dataset'] = cand
                break

    return config


def main():
    args = parse_args()

    # Setup automated file logging if not already captured by outer runner script
    if not os.environ.get("RUNNER_CAPTURING_LOG"):
        logs_dir = os.path.join(args['export_dir'], 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(logs_dir, f"train_{timestamp_str}.log")
        latest_log_path = os.path.join(logs_dir, "training.log")
        try:
            log_fh = open(log_file_path, "a", encoding="utf-8")
            sys.stdout = TeeStream(sys.stdout, log_fh)
            sys.stderr = TeeStream(sys.stderr, log_fh)
            print(f"[Logging] Phiên huấn luyện tự động ghi log tại: {log_file_path}")
            try:
                import shutil
                shutil.copyfile(log_file_path, latest_log_path)
            except Exception:
                pass
        except Exception as e:
            print(f"[Logging Warning] Không thể mở file log {log_file_path}: {e}")

    cfg = load_config(args['config'])

    # Áp dụng CLI overrides (nếu người dùng truyền vào từ dòng lệnh)
    if args['dataset_dir']:
        cfg['data']['path_to_dataset'] = args['dataset_dir']
        print(f"[Config Override] path_to_dataset -> {args['dataset_dir']}")

    if args['batch_size'] is not None:
        cfg['train']['batch_size'] = args['batch_size']
        print(f"[Config Override] batch_size -> {args['batch_size']}")

    if args['max_epoch'] is not None:
        cfg['train']['max_epoch'] = args['max_epoch']
        print(f"[Config Override] max_epoch -> {args['max_epoch']}")

    if args['devices'] is not None:
        dev_val = int(args['devices']) if args['devices'].isdigit() else args['devices']
        cfg['train']['devices'] = dev_val
        cfg['val']['devices'] = 1 if isinstance(dev_val, int) else dev_val
        print(f"[Config Override] devices -> {dev_val}")

    if args['num_workers'] is not None:
        cfg['data']['num_workers'] = args['num_workers']
        print(f"[Config Override] num_workers -> {args['num_workers']}")

    if args['check_val_every_n_epoch'] is not None:
        cfg['val']['check_val_every_n_epoch'] = args['check_val_every_n_epoch']
        print(f"[Config Override] check_val_every_n_epoch -> {args['check_val_every_n_epoch']}")

    if args.get('learning_rate') is not None:
        cfg['train']['learning_rate'] = args['learning_rate']
        print(f"[Config Override] learning_rate -> {args['learning_rate']}")

    if args.get('precision') is not None:
        cfg['train']['precision'] = args['precision']
        print(f"[Config Override] precision -> {args['precision']}")

    if args.get('scheduler') is not None:
        cfg['train']['scheduler'] = args['scheduler']
        print(f"[Config Override] scheduler -> {args['scheduler']}")

    if args.get('warmup_epochs') is not None:
        cfg['train']['warmup_epochs'] = args['warmup_epochs']
        print(f"[Config Override] warmup_epochs -> {args['warmup_epochs']}")

    if args.get('accumulate_grad_batches') is not None:
        cfg['train']['accumulate_grad_batches'] = args['accumulate_grad_batches']
        print(f"[Config Override] accumulate_grad_batches -> {args['accumulate_grad_batches']}")
    
    if cfg.get('seed') is None:
        seed_val = int(time.time())
        cfg['seed'] = seed_val
    else:
        seed_val = cfg['seed']
    seed_everything(seed_val) 

    datasetmodule = get_data_module(cfg) 
    criterion = get_criterion(cfg)

    # define backbone
    network = get_backbone(cfg)

    scheduler_type = cfg['train'].get('scheduler', 'poly')
    warmup_epochs = cfg['train'].get('warmup_epochs', 5)

    if (args['ckpt_path'] is not None) and (not args['resume']):
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                criterion,
                                                cfg['train']['learning_rate'],
                                                cfg['train']['weight_decay'],
                                                train_step_settings=cfg['train']['step_settings'],
                                                val_step_settings=cfg['val']['step_settings'],
                                                ckpt_path=args['ckpt_path'],
                                                scheduler_type=scheduler_type,
                                                warmup_epochs=warmup_epochs)
    else:
        seg_module = model_multimetrics.SegmentationNetwork(network,
                                                            criterion,
                                                            cfg['train']['learning_rate'],
                                                            cfg['train']['weight_decay'],
                                                            train_step_settings=cfg['train']['step_settings'],
                                                            val_step_settings=cfg['val']['step_settings'],
                                                            scheduler_type=scheduler_type,
                                                            warmup_epochs=warmup_epochs)

    # Add callbacks
    ckpt_dir = os.path.join(args['export_dir'], 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Guaranteed persistent last.ckpt saved on EVERY epoch end for pause/resume tolerance
    checkpoint_saver_epoch_last = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_top_k=0,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True
    )
    # Best mIoU checkpoint
    checkpoint_saver_val_mIoU = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_mIoU',
        filename=cfg['experiment']['id'] + '_best_mIoU_epoch{epoch:02d}_{val_mIoU:.4f}',
        mode='max',
        save_last=False
    )
    # Best Weed IoU checkpoint (Target Metric for Weed Segmentation)
    checkpoint_saver_val_weed_iou = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_weed_iou',
        filename=cfg['experiment']['id'] + '_best_weedIoU_epoch{epoch:02d}_{val_weed_iou:.4f}',
        mode='max',
        save_last=False
    )
    # Best validation loss checkpoint
    checkpoint_saver_val_loss = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_loss',
        filename=cfg['experiment']['id'] + '_best_loss_epoch{epoch:02d}_{val_loss:.4f}',
        mode='min',
        save_last=False
    )
    # Periodic checkpoint every 10 epochs
    checkpoint_saver_periodic = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=cfg['experiment']['id'] + '_periodic_epoch{epoch:04d}',
        every_n_epochs=10,
        save_top_k=-1,
        save_last=False,
        save_on_train_epoch_end=True
    )


    my_checkpoint_savers = [var_value for var_name, var_value in locals().items() if
                            var_name.startswith('checkpoint_saver')]

    visualizer_callback = VisualizerCallback(get_visualizers(cfg),
                                             cfg['train']['vis_train_every_x_epochs'],
                                             cfg['val']['vis_val_every_x_epochs'])

    postprocessor_callback = PostprocessorCallback(get_postprocessors(cfg),
                                                   cfg['train']['postprocess_train_every_x_epochs'],
                                                   cfg['val']['postprocess_val_every_x_epochs'])

    config_callback = ConfigCallback(cfg)

    all_callbacks = [
        *my_checkpoint_savers,
        lr_monitor,
        visualizer_callback,
        postprocessor_callback,
        config_callback,
    ]

    # Cấu hình EarlyStopping linh hoạt (Mặc định: PaperEarlyStopping theo chuẩn bài báo IEEE TGRS 2026: 3 lần val liên tiếp có val_loss <= 0.1)
    use_early_stopping = not args['no_early_stopping']
    es_cfg = cfg['train'].get('early_stopping', {})
    if isinstance(es_cfg, bool):
        es_cfg = {'mode': 'paper' if es_cfg else 'none'}

    es_mode = args.get('early_stopping_mode') or es_cfg.get('mode', 'paper')
    if args['no_early_stopping'] or es_mode == 'none':
        use_early_stopping = False

    if use_early_stopping:
        if es_mode == 'paper':
            threshold = args.get('early_stopping_threshold') if args.get('early_stopping_threshold') is not None else es_cfg.get('loss_threshold', 0.1)
            consecutive = args.get('early_stopping_consecutive') if args.get('early_stopping_consecutive') is not None else es_cfg.get('consecutive_count', 3)
            paper_es = PaperEarlyStopping(
                loss_threshold=threshold,
                consecutive_count=consecutive,
                monitor='val_loss',
                verbose=True
            )
            all_callbacks.append(paper_es)
            print(f"[EarlyStopping] Kích hoạt chuẩn Paper (IEEE TGRS): {consecutive} lần val liên tiếp có val_loss <= {threshold}")
        elif es_mode == 'plateau':
            patience = args.get('early_stopping_patience') if args.get('early_stopping_patience') is not None else es_cfg.get('patience', 20)
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=0.001,
                mode='min',
                verbose=True
            )
            all_callbacks.append(early_stopping)
            print(f"[EarlyStopping] Kích hoạt Plateau EarlyStopping với patience = {patience}")
    else:
        print("[EarlyStopping] Đã tắt, mô hình sẽ huấn luyện đủ max_epoch theo cấu hình paper.")
 
    # Automated Compute Profiling Callback for Task W1-T4 / W1-T5
    if args.get('profile', False):
        profiling_cb = ComputeProfilingCallback(
            warmup_batches=args.get('profile_warmup_batches', 10),
            target_max_epochs=4096,
            val_interval=200,
            output_dir=args['export_dir'],
            verbose=True
        )
        all_callbacks.append(profiling_cb)
        print("[Profiling] ComputeProfilingCallback kích hoạt cho W1-T5 Profiling Run.")

    # Đảm bảo export_dir tồn tại
    os.makedirs(args['export_dir'], exist_ok=True)

    # Setup strategy
    train_devices = cfg['train'].get('devices', 'auto')
    train_strategy = cfg['train'].get('strategy', 'auto')
    if train_strategy == 'auto' and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from pytorch_lightning.strategies import DDPStrategy
        train_strategy = DDPStrategy(find_unused_parameters=False)

    # Setup logger
    try:
        tb_logger = TensorBoardLogger(
            save_dir=args['export_dir'],
            name="lightning_logs",
            default_hp_metric=False
        )
    except Exception:
        tb_logger = True

    # Setup precision & gradient accumulation
    raw_precision = cfg['train'].get('precision', 32)
    if str(raw_precision) in ['16', '16-mixed', 'bf16']:
        try:
            train_precision = int(raw_precision) if str(raw_precision).isdigit() else raw_precision
        except Exception:
            train_precision = raw_precision
    else:
        train_precision = 32

    train_accum_grad = int(cfg['train'].get('accumulate_grad_batches', 1))

    # Setup trainer
    trainer = Trainer(
        accelerator=cfg['train'].get('accelerator', 'auto'),
        devices=train_devices,
        strategy=train_strategy,
        benchmark=cfg['train'].get('benchmark', True),
        precision=train_precision,
        accumulate_grad_batches=train_accum_grad,
        default_root_dir=args['export_dir'],
        max_epochs=cfg['train']['max_epoch'],
        check_val_every_n_epoch=cfg['val']['check_val_every_n_epoch'],
        callbacks=all_callbacks,
        logger=tb_logger)

    if args['ckpt_path'] is None:
        print('Train from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and (not args['resume']):
        print('Load pretrained model weights but other params (e.g. learning rate) start from scratch.')
        trainer.fit(seg_module, datasetmodule)
    elif (args['ckpt_path'] is not None) and args['resume']:
        print("Load pretrained model weights and resume training.")
        trainer.fit(seg_module, datasetmodule, ckpt_path=args['ckpt_path'])
    else:
        raise RuntimeError("Can't train any model since the settings are invalid.")

    # Generate validation curves and summary report automatically
    try:
        from utils.plot_val_curves import generate_curves_and_summary
        generate_curves_and_summary(args['export_dir'])
    except Exception as e:
        print(f"[Post-Train Notice] Automatic plot generation skipped: {e}")


if __name__ == '__main__':
    main()




