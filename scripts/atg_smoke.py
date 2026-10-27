"""Actual Lightning resume and actual CLI/PDC synthetic smoke, never campaign data."""
from pathlib import Path
from copy import deepcopy
from datetime import timedelta
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from models import get_backbone, get_criterion
from models.model_multimetrics import SegmentationNetwork
from callbacks.config_callback import ConfigCallback
from callbacks.atg_cost import ATGCostCallback
from scripts.export_atg import load_model
ROOT=Path(__file__).resolve().parents[1]

def module(cfg):
    return SegmentationNetwork(get_backbone(cfg),get_criterion(cfg),cfg['train']['learning_rate'],cfg['train']['weight_decay'],
        train_step_settings=['regular'],val_step_settings=['regular'],scheduler_type=cfg['train']['scheduler'],warmup_epochs=cfg['train']['warmup_epochs'])

class Synthetic(Dataset):
    def __len__(self):return 8
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i)
        return {'input_image':torch.randn(3,32,32,generator=g),'anno':torch.randint(0,3,(32,32),generator=g)}

def trainer_smoke(cfg,output,accelerator='cpu',devices=1,resume=False):
    pl.seed_everything(cfg['seed'])
    net=module(cfg);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    logger=TensorBoardLogger(str(output),name='lightning_logs',version=0,default_hp_metric=False)
    ckpt=ModelCheckpoint(dirpath=str(output/'checkpoints'),save_last=True,monitor='val_mIoU',mode='max')
    trainer=pl.Trainer(accelerator=accelerator,devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False,timeout=timedelta(seconds=90)) if devices>1 else 'auto',
        sync_batchnorm=False,precision=32,max_epochs=2 if resume else 1,
        limit_train_batches=2,limit_val_batches=1,num_sanity_val_steps=2,enable_progress_bar=False,
        enable_model_summary=False,log_every_n_steps=1,default_root_dir=str(output),logger=logger,
        callbacks=[ConfigCallback(cfg),ckpt,ATGCostCallback(output)])
    loader=DataLoader(Synthetic(),batch_size=2)
    trainer.fit(net,loader,loader,ckpt_path=str(output/'checkpoints/last.ckpt') if resume else None)
    if trainer.is_global_zero:
        state=torch.load(output/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert state['global_step']==(4 if resume else 2)
        import csv
        expected_epochs = [0, 1] if resume else [0]
        with (output/'val_history.csv').open() as stream: validation_rows=list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in validation_rows] == expected_epochs
        assert [row['epoch'] for row in state['val_history']] == expected_epochs
        assert net.val_history == state['val_history']
        assert state['optimizer_states'] and all(torch.isfinite(v).all() for v in state['state_dict'].values())
        load_model(cfg,output/'checkpoints/last.ckpt')
        cost=json.loads((output/'training_cost.json').read_text());assert cost['world_size']==devices
        assert cost['main_trainable_params']==sum(p.numel() for p in net.network.parameters())
        print(f'PASS trainer {cfg["experiment"]["id"]}: devices={devices}, resume={resume}',flush=True)

def pipeline_smoke(cfg, output, accelerator='cpu', devices=1):
    """Actual trainer CLI + PDC loader on explicitly synthetic32 crops/1 epoch."""
    import subprocess, sys
    from PIL import Image
    output=Path(output);dataset=output/'synthetic_data'
    rng=np.random.default_rng(cfg['seed'])
    for split,count in [('train',8),('val',4),('test',0)]:
        for folder in ['images','semantics']:(dataset/split/folder).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(dataset/split/'images'/f'synthetic_{i}.png')
            Image.fromarray(rng.integers(0,5,(32,32),dtype=np.uint8)).save(dataset/split/'semantics'/f'synthetic_{i}.png')
    smoke=deepcopy(cfg);smoke['data']['path_to_dataset']=str(dataset.resolve());smoke['data']['num_workers']=0
    smoke['train'].update(max_epoch=1,batch_size=2,accelerator=accelerator,devices=devices)
    if accelerator=='cpu' and devices>1:smoke['train']['strategy']='ddp'
    smoke['train']['geometric_data_augmentations']['random_crop']={'height':32,'width':32}
    smoke['val'].update(batch_size=2,check_val_every_n_epoch=1)
    smoke['val']['geometric_data_augmentations']['center_crop']={'height':32,'width':32}
    config_path=output/'synthetic_config.yaml';config_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
    subprocess.run([sys.executable,str(ROOT/'multi_metric_train.py'),'--config',str(config_path),
        '--export_dir',str(output/'training_output'),'--devices',str(devices)],cwd=ROOT,check=True)
    assert (output/'training_output/checkpoints/last.ckpt').is_file()
    assert (output/'training_output/val_history.csv').is_file()
    assert json.loads((output/'training_output/training_cost.json').read_text())['world_size']==devices
    from scripts.check_atg_run import audit_run
    audit=audit_run(output/'training_output',expected_epochs=1,val_interval=1)
    (output/'campaign_audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS actual trainer CLI + PDC loader, synthetic smoke only',flush=True)
