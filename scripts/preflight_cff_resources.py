"""Native real-data resource gate. OOM blocks campaign; never changes P2 recipe.

Two disposable train steps include Adam states, followed by native eval batch.
CPU --smoke-size is explicitly small-crop evidence, not a GPU resource claim.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from pytorch_lightning import seed_everything
from datasets import get_data_module
from models import get_backbone,get_criterion
from utils.cff_campaign import SCENARIOS,config


def measure(mode,dataset_dir,device,smoke_size=None):
    cfg=config(mode);cfg['data']['path_to_dataset']=str(Path(dataset_dir).resolve());cfg['data']['num_workers']=0
    if smoke_size is not None:
        if device!='cpu':raise ValueError('Native GPU resource gate does not accept size/batch overrides')
        cfg['train']['batch_size']=cfg['val']['batch_size']=2
        cfg['train']['geometric_data_augmentations']['random_crop']={'height':smoke_size,'width':smoke_size}
        cfg['val']['geometric_data_augmentations']['center_crop']={'height':smoke_size,'width':smoke_size}
    if cfg['train']['precision']!=32:raise ValueError('CFF NOW requires FP32')
    torch.backends.cudnn.benchmark=cfg['train'].get('benchmark',True)
    seed_everything(cfg['seed']);dm=get_data_module(cfg);dm.setup('fit')
    train_batch=next(iter(dm.train_dataloader()));val_batch=next(iter(dm.val_dataloader()))
    model=get_backbone(cfg).float().to(device).train();criterion=get_criterion(cfg).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay'])
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    x=train_batch['input_image'].to(device);labels=train_batch['anno'].to(device)
    losses=[]
    for step in range(2):
        print(f'{mode}: native train step {step+1}/2, input={list(x.shape)}',flush=True)
        optimizer.zero_grad(set_to_none=True);logits=model(x)
        loss=criterion(logits,labels,mode='train',mask_keep=labels!=255);loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite native train loss/gradient')
        optimizer.step();losses.append(float(loss.detach()));del logits,loss
    train_peak={'allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
                'reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None}
    optimizer.zero_grad(set_to_none=True);del x,labels
    # Keep Adam states allocated, as during real fit validation.
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    model.eval();vx=val_batch['input_image'].to(device);vy=val_batch['anno'].to(device)
    print(f'{mode}: native validation, input={list(vx.shape)}',flush=True)
    with torch.inference_mode():
        prediction=model(vx);vloss=criterion(prediction,vy,mode='val')
        if prediction.shape!=(vx.shape[0],3,*vy.shape[-2:]) or not torch.isfinite(vloss):raise FloatingPointError('Invalid native validation')
    return {'status':'passed','device':device,'evidence_type':'GPU_native_real_batch' if device=='cuda' else 'CPU_small_smoke_not_GPU_evidence',
            'train_input_shape':list(train_batch['input_image'].shape),'val_input_shape':list(vx.shape),
            'train_steps_with_Adam_states':2,'train_losses':losses,'val_loss':float(vloss),
            'train_peak':train_peak,'val_peak':{'allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None},
            'scope':'single-device disposable real batch; DDP overhead/throughput tested separately; no epoch training',
            'recipe_overrides':{'workers':0,'smoke_size':smoke_size}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--smoke-size',type=int)
    p.add_argument('--experiments',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS));args=p.parse_args()
    torch.set_num_threads(2);output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','results':{},'automatic_recipe_fallback':False,
            'native_protocol':{'train_batch_per_GPU':4,'train_crop':768,'val_batch_per_GPU':4,'val_crop':1024,'precision':32}}
    try:
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required for native resource acceptance')
        for mode in args.experiments:
            report['current_mode']=mode
            report['results'][mode]=measure(mode,args.dataset_dir,args.device,args.smoke_size)
            print('PASS native-resource '+mode,flush=True)
            if args.device=='cuda':torch.cuda.empty_cache()
        report['status']='passed'
    except torch.cuda.OutOfMemoryError as err:
        report.update(status='blocked_OOM',error=str(err),action='Stop and report to team; batch/crop/precision/topology unchanged')
        raise
    except Exception as err:
        report.update(status='failed',error=str(err));raise
    finally:
        output.write_text(json.dumps(report,indent=2)+'\n')
