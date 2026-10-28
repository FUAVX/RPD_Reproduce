"""Disposable paired execution with native real data on a single A100.

No batch/crop/precision fallback. CPU mode is explicitly synthetic/small evidence.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch,yaml
import pytorch_lightning as pl
from pytorch_lightning import Callback
from models import get_backbone,get_criterion
from utils.ndc_colab import ColabNetwork,ColabDataModule,PairedRankState,validate_recipe,configure_fp32

class ResourceProbe(Callback):
    def __init__(self):self.train_peak=None;self.val_peak=None;self.backward_checks=0
    def peak(self):return {'allocated_bytes':torch.cuda.max_memory_allocated(),'reserved_bytes':torch.cuda.max_memory_reserved()} if torch.cuda.is_available() else None
    def on_after_backward(self,trainer,module):
        for name,param in module.network.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():raise FloatingPointError('Invalid gradient '+name)
        self.backward_checks+=1
    def on_validation_start(self,trainer,module):
        self.train_peak=self.peak()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    def on_validation_end(self,trainer,module):self.val_peak=self.peak()

def run(cfg,output):
    validate_recipe(cfg);configure_fp32();pl.seed_everything(cfg['seed'])
    cpu=cfg['train'].get('accelerator')=='cpu'
    if not cpu and (not torch.cuda.is_available() or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0)):
        raise RuntimeError('Select a single A100 runtime; no hardware fallback')
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    model=ColabNetwork(get_backbone(cfg),get_criterion(cfg),cfg['train']['learning_rate'],cfg['train']['weight_decay'],
        train_step_settings=['regular'],val_step_settings=['regular'],scheduler_type=cfg['train']['scheduler'],warmup_epochs=cfg['train']['warmup_epochs'])
    probe=ResourceProbe();dm=ColabDataModule(cfg)
    trainer=pl.Trainer(accelerator='cpu' if cpu else 'gpu',devices=1,precision=32,max_epochs=1,
        accumulate_grad_batches=2,sync_batchnorm=False,benchmark=cfg['train']['benchmark'],
        limit_train_batches=4,limit_val_batches=2,num_sanity_val_steps=0,
        enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,
        logger=False,default_root_dir=str(Path(output).parent/'disposable'),callbacks=[PairedRankState(),probe])
    trainer.fit(model,dm)
    if trainer.global_step!=2 or probe.backward_checks!=4:raise RuntimeError('Paired optimizer-step count mismatch')
    return {'status':'passed','protocol':cfg['execution']['protocol'],'evidence':'CPU_smoke_only' if cpu else 'A100_native_real_data',
            'optimizer_steps':trainer.global_step,'microbatches':probe.backward_checks,'batch_per_micro':4,'effective_batch':8,
            'train_crop_requested':cfg['train']['geometric_data_augmentations']['random_crop'],
            'train_peak':probe.train_peak,'val_peak':probe.val_peak,'TF32':False,
            'note':'disposable2Adamsteps +2valbatches; not campaign training or quality evidence'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    try:result=run(yaml.safe_load(Path(args.config).read_text()),out)
    except torch.cuda.OutOfMemoryError as e:result={'status':'blocked_OOM','error':str(e)};raise
    except Exception as e:result={'status':'failed','error':str(e)};raise
    finally:out.write_text(json.dumps(result,indent=2)+'\n')
