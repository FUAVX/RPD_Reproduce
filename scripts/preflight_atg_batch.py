"""One real PDC batch: production augmentation/WDice, finite backward, isolated process."""
import argparse
from copy import deepcopy
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
from utils.atg_campaign import SCENARIOS,config


def check(mode,dataset_dir,device='cpu',smoke_size=None):
    cfg=config(mode);cfg['data']['path_to_dataset']=str(Path(dataset_dir).resolve())
    cfg['data']['num_workers']=0  # one disposable batch, no persistent loader workers
    if smoke_size is not None:
        cfg['train']['batch_size']=2
        cfg['train']['geometric_data_augmentations']['random_crop']={'height':smoke_size,'width':smoke_size}
    seed_everything(cfg['seed'])
    dm=get_data_module(cfg);dm.setup('fit');batch=next(iter(dm.train_dataloader()))
    x=batch['input_image'].to(device);labels=batch['anno'].to(device)
    model=get_backbone(cfg).to(device).train();criterion=get_criterion(cfg).to(device)
    output=model(x);loss=criterion(output,labels,mode='train',mask_keep=labels!=255)
    assert output.shape==(x.shape[0],3,*labels.shape[-2:]) and torch.isfinite(loss)
    loss.backward()
    for name,p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise RuntimeError(f'{mode}: invalid gradient {name}')
    return {'status':'passed','dataset_path':cfg['data']['path_to_dataset'],'input_shape':list(x.shape),
            'filenames':list(batch['fname']),'loss':float(loss.detach()),'finite_gradients':True,
            'label_ids':torch.unique(labels).cpu().tolist(),'scope':'one real training batch, no optimizer update; independent process',
            'device':device,'smoke_size_override':smoke_size}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--output',required=True)
    p.add_argument('--experiments',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS))
    p.add_argument('--smoke-size',type=int,help='CPU-only small crop evidence, never the full campaign')
    args=p.parse_args();torch.set_num_threads(2)
    results={}
    for mode in args.experiments:
        results[mode]=check(mode,args.dataset_dir,args.device,args.smoke_size)
        print('PASS real batch '+mode,flush=True)
        if args.device=='cuda':torch.cuda.empty_cache()
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(results,indent=2)+'\n')
