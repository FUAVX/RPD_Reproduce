"""Executed-convolution compute and native-shape checks for the locked Fuse modes."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from models import get_backbone
from utils.fpn_campaign import SCENARIOS,config


def profile(mode,device='cpu',sizes=((768,768),(1024,1024),(33,35))):
    cfg=config(mode);torch.manual_seed(cfg['seed'])
    model=get_backbone(cfg).to(device).eval()
    record={'trainable_params':sum(p.numel() for p in model.parameters()),'graph':'unfused_single_head',
            'fusion':deepcopy(cfg['backbone'].get('fusion')),'shapes':[]}
    for index,(h,w) in enumerate(sizes):
        x=torch.randn(1,3,h,w,device=device)
        with torch.inference_mode():
            if index==0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
                    y=model(x)
                flops=int(sum(e.flops for e in prof.key_averages() if e.key=='aten::conv2d'))
                record.update(conv_macs=flops//2,conv_flops=flops,compute_input_shape=list(x.shape))
            else:y=model(x)
        assert y.shape==(1,3,h,w) and torch.isfinite(y).all()
        record['shapes'].append({'input':list(x.shape),'output':list(y.shape),'finite':True})
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--experiments',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS));p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2)
    records={}
    for mode in args.experiments:
        records[mode]=profile(mode,args.device)
        print(mode,records[mode]['trainable_params'],records[mode]['conv_macs'],flush=True)
        if args.device=='cuda':torch.cuda.empty_cache()
    report={'status':'passed','torch':torch.__version__,'device':args.device,'precision':'FP32','results':records,
            'compute_scope':'executed aten::conv2d only, batch1/768, unfused; FLOPs=2MACs; excludes pool/add/resize/norm/activation/backward/optimizer'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
