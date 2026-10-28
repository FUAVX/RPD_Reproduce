"""Executed unfused Conv2d MACs + native forward shapes, not training memory."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
from models import get_backbone
from models.rpdnet.RPD_ops import Conv2d as RPDConv2d
from utils.cff_campaign import SCENARIOS,config

def profile(mode,device='cpu',sizes=((768,768),(1024,1024),(33,35))):
    cfg=config(mode);torch.manual_seed(cfg['seed']);model=get_backbone(cfg).to(device).eval()
    record={'trainable_params':sum(p.numel() for p in model.parameters()),'graph':'unfused_single_head','cff':cfg['backbone'].get('cff'),'shapes':[]}
    macs=0
    def count(module,inputs,output):
        nonlocal macs
        if isinstance(module,RPDConv2d) and module.op not in (None,torch.nn.functional.conv2d) and getattr(module.op,'op_type',None)!='cv':
            raise ValueError('CFF host profiling expects vanilla RPD cv operators')
        kh,kw=module.weight.shape[-2:]
        macs+=output.numel()*(module.in_channels//module.groups)*kh*kw
    for i,(h,w) in enumerate(sizes):
        hooks=[m.register_forward_hook(count) for m in model.modules() if isinstance(m,(nn.Conv2d,RPDConv2d))] if i==0 else []
        try:
            with torch.inference_mode():y=model(torch.randn(1,3,h,w,device=device))
        finally:
            for hook in hooks:hook.remove()
        assert y.shape==(1,3,h,w) and torch.isfinite(y).all()
        record['shapes'].append({'input':[1,3,h,w],'output':list(y.shape),'finite':True})
        if i==0:record.update(conv_macs=macs,conv_flops=2*macs,compute_input_shape=[1,3,h,w])
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--experiments',nargs='+',choices=list(SCENARIOS),default=list(SCENARIOS));p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(2);records={}
    for mode in args.experiments:
        records[mode]=profile(mode,args.device);print(mode,records[mode]['trainable_params'],records[mode]['conv_macs'],flush=True)
        if args.device=='cuda':torch.cuda.empty_cache()
    report={'status':'passed','torch':str(torch.__version__),'device':args.device,'precision':'FP32','results':records,
            'compute_scope':'executed nn.Conv2d + functional vanilla RPDConv2d, batch1/768, unfused; FLOPs=2MACs; excludes pool/add/resize/norm/activation/backward/optimizer'}
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
