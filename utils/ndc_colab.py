"""NDC single-A100 execution adapter; model/loss/optimizer remain P2.

Paired batch4 forwards reproduce two-rank mean gradients for fixed inputs and
RNG, retaining rank0 BN buffers (DDP broadcast_buffers=True semantics). This
is not bitwise replay of historical CUDA/data-worker streams.
"""
import math
import torch
from torch import nn
from torch.utils.data import DataLoader,Sampler
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets.pdc import PDCModule
from models.model_multimetrics import SegmentationNetwork

from utils.ndc_colab_io import PROTOCOL,validate_recipe,publish_backup,restore_backup

def configure_fp32():
    if hasattr(torch.backends,'fp32_precision'):
        torch.backends.fp32_precision='ieee'
        torch.backends.cuda.matmul.fp32_precision='ieee'
        torch.backends.cudnn.fp32_precision='ieee'
    else:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')

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

class ColabLastCheckpoint(ModelCheckpoint):
    def __init__(self,*args,backup_dir,**kwargs):super().__init__(*args,**kwargs);self.backup_dir=backup_dir
    def _save_checkpoint(self,trainer,filepath):
        super()._save_checkpoint(trainer,filepath)
        # Cost/log files may still be open; checkpoint/history are authoritative.
        publish_backup(trainer.default_root_dir,self.backup_dir)
