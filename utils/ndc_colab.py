"""NDC single-A100 execution adapter; model/loss/optimizer remain P2.

Paired batch4 forwards reproduce two-rank mean gradients for fixed inputs and
RNG, retaining rank0 BN buffers (DDP broadcast_buffers=True semantics). This
is not bitwise replay of historical CUDA/data-worker streams.
"""
from copy import deepcopy
import hashlib,json,math,os,random,shutil,tempfile,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Sampler
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.pdc import PDCModule
from models.model_multimetrics import SegmentationNetwork

PROTOCOL='ndc_colab_paired_batch4_v1'

def configure_fp32():
    if hasattr(torch.backends,'fp32_precision'):
        torch.backends.fp32_precision='ieee'
        torch.backends.cuda.matmul.fp32_precision='ieee'
        torch.backends.cudnn.fp32_precision='ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')

def validate_recipe(cfg):
    from utils.ndc_campaign import config
    mode='control' if cfg['backbone'].get('nested') is None else 'ex1'
    expected=config(mode);actual=deepcopy(cfg);execution=actual.pop('execution')
    if execution['protocol']!=PROTOCOL:raise ValueError('Unknown Colab execution protocol')
    for key in ['path_to_dataset','num_workers']:
        expected['data'][key]=actual['data'][key]
    expected['train']['devices']=1;expected['train']['accumulate_grad_batches']=2
    expected['train']['accelerator']=actual['train'].get('accelerator','gpu')
    if execution.get('smoke_evidence_only'):
        if expected['train']['accelerator']!='cpu':raise ValueError('Smoke overrides are CPU-only')
        for key in ['max_epoch']:expected['train'][key]=actual['train'][key]
        expected['train']['geometric_data_augmentations']['random_crop']=actual['train']['geometric_data_augmentations']['random_crop']
        expected['val']['check_val_every_n_epoch']=actual['val']['check_val_every_n_epoch']
    if actual!=expected:raise ValueError('Colab recipe drift outside approved device/paired execution/data-path overrides')
    if cfg['train']['devices']!=1 or cfg['train']['accumulate_grad_batches']!=2:raise ValueError('Single GPU +2 paired microbatches required')
    return mode

class PairedSampler(Sampler):
    """Pad/shard like DistributedSampler(2), interleave rank0/1 local batches.

    Train1407 -> pad1408 ->352 microbatches ->176 optimizer steps.
    Val772 ->386/rank ->drop2/rank ->768 images, same local groups as DDP.
    """
    def __init__(self,dataset,batch_size=4,shuffle=True,seed=1682409321):
        self.size=len(dataset);self.batch_size=batch_size;self.shuffle=shuffle;self.seed=seed;self.epoch=0
        self.per_rank=math.ceil(self.size/2);self.used=(self.per_rank//batch_size)*batch_size
        if not self.used:raise ValueError('Dataset too small for two complete local batches')
    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return self.used*2
    def __iter__(self):
        g=torch.Generator().manual_seed(self.seed+self.epoch)
        indices=torch.randperm(self.size,generator=g).tolist() if self.shuffle else list(range(self.size))
        indices+=(indices*math.ceil((2*self.per_rank-self.size)/self.size))[:2*self.per_rank-self.size]
        ranks=[indices[r:2*self.per_rank:2][:self.used] for r in range(2)]
        for i in range(0,self.used,self.batch_size):
            for rank in ranks:yield from rank[i:i+self.batch_size]

class ColabDataModule(PDCModule):
    def setup(self,stage=None):
        super().setup(stage)
        if not self.cfg['execution'].get('smoke_evidence_only') and (len(self.train_ds),len(self.val_ds))!=(1407,772):
            raise ValueError('Expected original PhenoBench train1407/val772; audit before changing sampling')
    def _loader(self,dataset,shuffle):
        workers=self.cfg['data']['num_workers'];seed=self.cfg['seed']
        return DataLoader(dataset,batch_size=4,sampler=PairedSampler(dataset,shuffle=shuffle,seed=seed),
                          drop_last=True,num_workers=workers,pin_memory=True,persistent_workers=workers>0,
                          generator=torch.Generator().manual_seed(seed))
    def train_dataloader(self):return self._loader(self.train_ds,True)
    def val_dataloader(self):return self._loader(self.val_ds,False)

class PairedRankState(Callback):
    """Keep two independent Dropout RNG streams and DDP rank0 BN buffers."""
    def __init__(self):self.streams=None;self.pre_bn=None;self.rank0_bn=None
    @property
    def state_key(self):return PROTOCOL+'.rank_state'
    def _rng(self):
        return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state() if torch.cuda.is_available() else None}
    def _set_rng(self,state):
        torch.set_rng_state(state['cpu'])
        if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'])
    def _bn(self,net):
        return {name:buf.detach().clone() for name,buf in net.named_buffers() if name.endswith(('running_mean','running_var','num_batches_tracked'))}
    def _set_bn(self,net,state):
        with torch.no_grad():
            for name,buf in net.named_buffers():
                if name in state:buf.copy_(state[name])
    def on_train_epoch_start(self,trainer,pl_module):
        trainer.train_dataloader.sampler.set_epoch(trainer.current_epoch)
        if len(trainer.train_dataloader)%2:raise ValueError('Odd number of microbatches would change effective batch')
        self.pre_bn=None;self.rank0_bn=None
    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        if self.streams is None:self.streams=[self._rng(),self._rng()]
        rank=batch_idx%2
        if rank==0:self.pre_bn=self._bn(pl_module.network)
        else:self._set_bn(pl_module.network,self.pre_bn)
        self._set_rng(self.streams[rank])
    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        rank=batch_idx%2;self.streams[rank]=self._rng()
        if rank==0:self.rank0_bn=self._bn(pl_module.network)
        else:self._set_bn(pl_module.network,self.rank0_bn)
    def state_dict(self):return {'streams':self.streams}
    def load_state_dict(self,state_dict):self.streams=state_dict['streams']

class ColabNetwork(SegmentationNetwork):
    def on_save_checkpoint(self,checkpoint):
        super().on_save_checkpoint(checkpoint);checkpoint['execution_protocol']=PROTOCOL
    def on_load_checkpoint(self,checkpoint):
        if checkpoint.get('execution_protocol')!=PROTOCOL:raise ValueError('Cannot resume a different execution protocol')
        super().on_load_checkpoint(checkpoint)

def publish_backup(source,backup_dir):
    """Two rotating ZIPs; publish pointer only after a complete checksum-verified copy."""
    source=Path(source);backup=Path(backup_dir);backup.mkdir(parents=True,exist_ok=True)
    source_commit=(source/'source_commit.txt').read_text().strip()
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Backup must record a valid exact source commit SHA')
    pointer=backup/'latest.json'
    previous=json.loads(pointer.read_text()) if pointer.exists() else {}
    slot='B' if previous.get('slot')=='A' else 'A'
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(shutil.make_archive(str(Path(tmp)/'snapshot'),'zip',root_dir=source))
        digest=hashlib.sha256(archive.read_bytes()).hexdigest();target=backup/f'snapshot_{slot}.zip'
        shutil.copyfile(archive,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise IOError('Drive backup checksum mismatch')
    meta={'slot':slot,'archive':target.name,'sha256':digest,'protocol':PROTOCOL,
          'source_commit':source_commit}
    temporary=backup/'latest.json.tmp';temporary.write_text(json.dumps(meta,indent=2)+'\n');os.replace(temporary,pointer)
    return meta

def restore_backup(backup_dir,destination):
    backup=Path(backup_dir);meta=json.loads((backup/'latest.json').read_text())
    if meta['protocol']!=PROTOCOL:raise ValueError('Backup protocol mismatch')
    archive=backup/meta['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['sha256']:raise IOError('Backup checksum mismatch')
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Restore destination must be empty')
    with zipfile.ZipFile(archive) as stream:
        for name in stream.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid archive path')
        stream.extractall(destination)
    return meta

class ColabLastCheckpoint(ModelCheckpoint):
    def __init__(self,*args,backup_dir,**kwargs):super().__init__(*args,**kwargs);self.backup_dir=backup_dir
    def _save_checkpoint(self,trainer,filepath):
        super()._save_checkpoint(trainer,filepath)
        # Cost/log files may still be open; checkpoint/history are authoritative.
        publish_backup(trainer.default_root_dir,self.backup_dir)
