#!/usr/bin/env python3
"""Disposable ATG benchmark; FP32, P2 Adam/WDice, no campaign mutation.

Unified project measurement: training20/100, inference30/200 warmup/timed.
These are benchmark settings, not model/training-campaign hyperparameters.
CPU/size/batch overrides are explicitly smoke evidence only.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml
from models import get_backbone, get_criterion

from utils.atg_campaign import SCENARIOS

@contextmanager
def isolated_rng(device):
    python_state=random.getstate();numpy_state=np.random.get_state();backend_benchmark=torch.backends.cudnn.benchmark
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
        torch.backends.cudnn.benchmark=backend_benchmark

def measure(cfg,args,device,world_size):
    with isolated_rng(device):
        torch.manual_seed(cfg['seed'])
        torch.backends.cudnn.benchmark=cfg['train'].get('benchmark',True)
        model=get_backbone(cfg).float().to(device)
        train_params=sum(p.numel() for p in model.parameters())
        # ATG training/inference share one graph; no auxiliary head to strip.
        params=sum(p.numel() for p in model.parameters())
        x=torch.randn(args.batch_size,3,args.size,args.size,device=device)
        target=torch.randint(0,3,(args.batch_size,args.size,args.size),device=device)
        criterion=get_criterion(cfg).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay']) if args.mode=='training' else None
        model.train(args.mode=='training')
        deep=None
        if world_size>1:model=DistributedDataParallel(model,device_ids=[device.index] if device.type=='cuda' else None,find_unused_parameters=False)
        def step():
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(x),target,mode='train',mask_keep=target!=255)
                loss.backward();optimizer.step()
                return loss.detach()
            with torch.inference_mode():return model(x)
        def sync():
            if device.type=='cuda':torch.cuda.synchronize(device)
        try:
            for _ in range(args.warmup):step()
            sync()
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for _ in range(args.iterations):
                sync();start=time.perf_counter();out=step();sync();elapsed=time.perf_counter()-start
                if not torch.isfinite(out).all():raise FloatingPointError('Non-finite disposable benchmark output')
                if world_size>1:
                    t=torch.tensor(elapsed,dtype=torch.float64,device=device);dist.all_reduce(t,op=dist.ReduceOp.MAX);elapsed=float(t)
                times.append(elapsed*1000)
            local={'rank':dist.get_rank() if world_size>1 else 0,
                   'allocated_MiB':torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
                   'reserved_MiB':torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None}
            memory=[None]*world_size
            if world_size>1:dist.all_gather_object(memory,local)
            else:memory=[local]
            mean=float(np.mean(times))
            return {'mean_ms':mean,'P50_ms':float(np.percentile(times,50)),'P95_ms':float(np.percentile(times,95)),
                    'images_per_second':args.batch_size*world_size*1000/mean,'trainable_params_in_measured_graph':params,
                    'training_model_params':train_params,'peak_memory_by_rank':memory,'resolution':[args.size,args.size],
                    'batch_per_rank':args.batch_size,'world_size':world_size,'warmup':args.warmup,'measured':args.iterations,
                    'timing':'synchronized host wall time; distributed maximum per iteration','graph':'training_with_aux' if deep and optimizer is not None else 'main_only',
                    'cudnn_benchmark':torch.backends.cudnn.benchmark,'scope':'synthetic disposable initialization/Adam; no dataloader, no checkpoint accuracy evidence'}
        finally:
            del model,optimizer,criterion,x,target
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':
    import os
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['training','inference'],required=True);parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--size',type=int,default=768);parser.add_argument('--batch-size',type=int)
    parser.add_argument('--warmup',type=int);parser.add_argument('--iterations',type=int)
    parser.add_argument('--experiments',nargs='+')
    parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    scenarios = SCENARIOS
    config_dir = 'configs/p3_fuse_atg'
    args.experiments = args.experiments or list(scenarios)
    if any(key not in scenarios for key in args.experiments) or 'control' not in args.experiments:
        parser.error('Select valid campaign modes and include control for relative timing')
    args.experiments = ['control'] + [k for k in dict.fromkeys(args.experiments) if k != 'control']
    args.batch_size=args.batch_size or (4 if args.mode=='training' else 1)
    args.warmup=args.warmup if args.warmup is not None else (20 if args.mode=='training' else 30)
    args.iterations=args.iterations if args.iterations is not None else (100 if args.mode=='training' else 200)
    if args.size<32 or min(args.batch_size,args.warmup,args.iterations)<1:parser.error('size>=32; batch/warmup/iterations positive')
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('LOCAL_RANK','0'))
    if args.mode=='inference' and world>1:parser.error('Inference measurement uses batch1, single GPU')
    device=torch.device(f'cuda:{rank}' if args.device=='cuda' else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    if device.type=='cpu':torch.set_num_threads(2)
    report={'mode':args.mode,'torch':torch.__version__,'precision':'FP32','world_size':world,
            'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'host_torch_num_threads':torch.get_num_threads(),'status':'running','evidence_type':'GPU_measurement' if device.type=='cuda' else 'CPU_smoke_not_GPU_evidence','results':{}}
    try:
        base=Path(__file__).resolve().parents[1]
        for key in args.experiments:
            cfg=yaml.safe_load((base/config_dir/f'{scenarios[key]}.yaml').read_text())
            report['results'][key]=measure(cfg,args,device,world)
            report['results'][key]['latency_delta_pct_vs_control']=100*(report['results'][key]['mean_ms']/report['results']['control']['mean_ms']-1) if 'control' in report['results'] else None
        report['status']='completed'
    finally:
        if world==1 or dist.get_rank()==0:
            path=Path(args.output_dir);path.mkdir(parents=True,exist_ok=True)
            (path/f'benchmark_{args.mode}.json').write_text(json.dumps(report,indent=2)+'\n')
        if world>1:dist.destroy_process_group()
