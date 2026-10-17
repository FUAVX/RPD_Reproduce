"""Regression checks for DDP config export; run `python test_config_callback.py`."""

import importlib.util
import os
import tempfile
from datetime import timedelta
from pathlib import Path

from utils.tensorboard_compat import use_tensorboard_without_tensorflow

use_tensorboard_without_tensorflow()

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, TensorDataset

# Load this callback without importing unrelated visualization dependencies.
spec = importlib.util.spec_from_file_location(
    'config_callback_test_subject', Path(__file__).parent / 'callbacks/config_callback.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ConfigCallback = module.ConfigCallback


def test_rank_zero_and_existing_directory():
    class TrainerStub:
        def __init__(self, path, rank_zero):
            self.path = path
            self.is_global_zero = rank_zero
            self.log_dir_reads = 0

        @property
        def log_dir(self):
            self.log_dir_reads += 1
            return str(self.path)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'logs'
        cfg = {'train': {'max_epoch': 100, 'devices': 2}}
        callback = ConfigCallback(cfg)
        worker = TrainerStub(path, False)
        callback.setup(worker, None)
        assert worker.log_dir_reads == 1, 'Every rank must resolve the potentially collective log_dir.'
        assert not path.exists(), 'Nonzero ranks must not create/write the config directory.'
        leader = TrainerStub(path, True)
        callback.setup(leader, None)
        callback.setup(leader, None)  # Existing directory must be safe.
        target = path / 'configuration/config.yaml'
        assert yaml.safe_load(target.read_text()) == cfg
        before = target.stat().st_mtime_ns
        callback.setup(worker, None)
        assert target.stat().st_mtime_ns == before, 'Nonzero ranks must not rewrite config.'
    print('PASS: collective path access, rank-zero-only export, existing directory')


class TinyNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(2, 1)

    def training_step(self, batch, batch_idx):
        loss = self.network(batch[0]).square().mean()
        self.log('train_smoke_loss', loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def run_two_process_training(directory):
    cfg = {'experiment': {'id': 'config_callback_ddp_regression'}, 'runtime_devices': 2}
    logger = TensorBoardLogger(str(directory), name='lightning_logs', version=0, default_hp_metric=False)
    checkpoint = ModelCheckpoint(dirpath=str(directory / 'checkpoints'), save_last=True)
    trainer = pl.Trainer(
        accelerator='cpu', devices=2,
        strategy=DDPStrategy(process_group_backend='gloo', timeout=timedelta(seconds=30),
                             find_unused_parameters=False),
        max_epochs=1, logger=logger, callbacks=[ConfigCallback(cfg), checkpoint],
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, log_every_n_steps=1)
    # Read the logger path directly, not the collective Trainer property before setup.
    expected_log_dir = Path(logger.log_dir)
    loader = DataLoader(TensorDataset(torch.randn(8, 2)), batch_size=2)
    trainer.fit(TinyNetwork(), loader)
    if trainer.is_global_zero:
        assert yaml.safe_load((expected_log_dir / 'configuration/config.yaml').read_text()) == cfg
        assert (directory / 'checkpoints/last.ckpt').is_file()
        assert trainer.global_step > 0
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(expected_log_dir)).Reload()
        assert events.Scalars('train_smoke_loss')
        print('PASS: 2-process DDP completed config export, real TensorBoard logging, training and checkpoint saving')


if __name__ == '__main__':
    torch.set_num_threads(2)
    # Lightning's subprocess launcher re-executes this script. Share the same
    # test output path across ranks, matching the Kaggle launcher's behavior.
    inherited = os.environ.get('P3_CONFIG_CALLBACK_TEST_DIR')
    if inherited:
        run_two_process_training(Path(inherited))
    else:
        test_rank_zero_and_existing_directory()
        with tempfile.TemporaryDirectory() as directory:
            os.environ['P3_CONFIG_CALLBACK_TEST_DIR'] = directory
            try:
                run_two_process_training(Path(directory))
            finally:
                os.environ.pop('P3_CONFIG_CALLBACK_TEST_DIR', None)
