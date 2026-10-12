"""Unit checks for fresh guards and manifest lifecycle; no real GPU training."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
import yaml
from utils.ende_training import FreshEnDeCallback, check_dataset, validate_fresh_config

ROOT = Path(__file__).resolve().parents[1]


class FreshContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load((ROOT/'configs/p2_scenarios/P2_EncoderHalf.yaml').read_text())

    def test_recipe_and_resume_guards(self):
        self.assertEqual(len(validate_fresh_config(self.cfg, {})), 5)
        for args in ({'resume': True}, {'ckpt_path': 'old.ckpt'}, {'early_stopping_mode': 'paper'}):
            with self.assertRaises(ValueError):
                validate_fresh_config(self.cfg, args)
        for key, value in [('devices', 1), ('batch_size', 8), ('precision', 16),
                           ('max_epoch', 99), ('drop_last', False), ('resume_checkpoint', 'old.ckpt')]:
            cfg = deepcopy(self.cfg)
            cfg['train'][key] = value
            with self.assertRaises(ValueError):
                validate_fresh_config(cfg, {})
        cfg = deepcopy(self.cfg)
        cfg['train']['geometric_data_augmentations']['random_crop']['height'] = 512
        with self.assertRaises(ValueError):
            validate_fresh_config(cfg, {})

    def test_missing_mask_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ['train', 'val']:
                (root/split/'images').mkdir(parents=True)
                (root/split/'semantics').mkdir()
                (root/split/'images'/'sample.png').touch()
            cfg = deepcopy(self.cfg)
            cfg['data']['path_to_dataset'] = temp
            with self.assertRaises(FileNotFoundError):
                check_dataset(cfg)

    def test_manifest_complete_failed_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = deepcopy(self.cfg)
            network = torch.nn.Linear(2, 3)
            network.channel_table = dict(stem=16)
            module = SimpleNamespace(network=network, val_history=[])
            trainer = SimpleNamespace(is_global_zero=True, global_rank=0, world_size=2, num_devices=2,
                accelerator=type('CUDAAccelerator', (), {})(), local_rank=0,
                current_epoch=0, global_step=0, accumulate_grad_batches=1, precision=32,
                optimizers=[torch.optim.Adam(network.parameters(), lr=2e-4)],
                strategy=SimpleNamespace(broadcast=lambda x: x), checkpoint_callbacks=[],
                datamodule=SimpleNamespace(train_ds=list(range(10)), val_ds=list(range(772)),
                    train_dataloader=lambda: DataLoader(range(10), batch_size=4, drop_last=True),
                    val_dataloader=lambda: DataLoader(range(772), batch_size=4, drop_last=True)))
            callback = FreshEnDeCallback(cfg, {'export_dir': temp}, {})
            with patch.dict(os.environ, WORLD_SIZE='2', LOCAL_RANK='0'), patch(
                    'utils.ende_training.check_dataset', return_value={'train':10, 'val':772}):
                callback.on_fit_start(trainer, module)
                manifest_path = Path(temp)/'experiment_manifest.json'
                self.assertEqual(json.loads(manifest_path.read_text())['dataset']['processed_val_images'],768)
                # Directory reuse must preserve the existing manifest.
                previous = manifest_path.read_text()
                with self.assertRaises(FileExistsError):
                    callback.on_fit_start(trainer, module)
                self.assertEqual(previous, manifest_path.read_text())
            trainer.current_epoch = 99
            callback.on_train_epoch_end(trainer, module)
            module.val_history = [{'epoch':i} for i in range(4,100,5)]
            callback.on_fit_end(trainer, module)
            self.assertEqual(json.loads(manifest_path.read_text())['status'],'complete')
            module.val_history = [{'epoch':4}]
            with self.assertRaises(RuntimeError):
                callback.on_fit_end(trainer, module)
            self.assertEqual(json.loads(manifest_path.read_text())['status'],'failed')
            callback.on_exception(trainer, module, OSError('mask I/O failed'))
            self.assertEqual(json.loads(manifest_path.read_text())['failure_reason'],'mask I/O failed')


if __name__ == '__main__':
    unittest.main()
