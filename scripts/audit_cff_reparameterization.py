"""Optional full-model RepDW fusion audit; default artifacts stay unfused.

Does not export weights, fold the new CFF BN, or relax FP32 tolerance.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch,yaml
from models.rpdnet.RPD_Module import RPD_model_deploy
from scripts.export_cff import load_model

def audit(model,sizes=((768,768),(1024,1024),(33,35)),seed=1682409321):
    original=model.eval();fused=RPD_model_deploy(original,do_copy=True).eval()
    from models.rpdnet.fuse_cff import CascadeFeatureFusion
    count=lambda net:sum(isinstance(m,CascadeFeatureFusion) for m in net.modules())
    result={'status':'passed','atol':1e-4,'rtol':1e-4,'CFF_nodes_expected':count(original),'CFF_nodes_preserved':count(fused),
            'cff_preserved':getattr(original,'cff',None)==getattr(fused,'cff',None),'probes':[],
            'scope':'optional RepDW/RPD conversion of full model; new CFF Conv/BN retained; no fused weights exported'}
    if result['CFF_nodes_expected']!=result['CFF_nodes_preserved'] or not result['cff_preserved']:result['status']='not_certified'
    device=next(original.parameters()).device;generator=torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for h,w in sizes:
            x=torch.randn(1,3,h,w,generator=generator).to(device);a=original(x);b=fused(x)
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            passed=finite and torch.allclose(a,b,rtol=1e-4,atol=1e-4)
            result['probes'].append({'shape':list(x.shape),'finite':finite,'passed':passed,'max_abs_error':float((a-b).abs().max()),
                                    'max_abs_reference_logits':float(a.abs().max()),'argmax_agreement':float((a.argmax(1)==b.argmax(1)).float().mean())})
            if not passed:result['status']='not_certified'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','report']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args();torch.set_num_threads(2)
    cfg=yaml.safe_load(Path(args.config).read_text());result=audit(load_model(cfg,args.checkpoint,args.device),seed=cfg['seed'])
    path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit('Fusion NOT certified; keep unfused model')
