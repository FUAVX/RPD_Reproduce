"""Optional full-model fusion certification. Never relax tolerance or export weights.

Default NDC artifacts remain unfused; failed audit is recorded and exits nonzero.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch,yaml
from models.rpdnet.RPD_Module import RPD_model_deploy
from scripts.export_ndc import load_model


def audit(model,sizes=((768,768),(1024,1024),(33,35)),seed=1682409321):
    original=model.eval();fused=RPD_model_deploy(original,do_copy=True).eval()
    device=next(original.parameters()).device;generator=torch.Generator().manual_seed(seed)
    result={'status':'passed','atol':1e-4,'rtol':1e-4,'H_nodes_preserved':len(getattr(fused,'h_units',{})),'probes':[],
            'scope':'full-model FP32 fusion; no weights exported; not an accuracy reevaluation'}
    result['H_nodes_expected']=len(getattr(original,'h_units',{}))
    if result['H_nodes_preserved']!=result['H_nodes_expected']:result['status']='not_certified'
    with torch.inference_mode():
        for h,w in sizes:
            x=torch.randn(1,3,h,w,generator=generator).to(device);a=original(x);b=fused(x)
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            passed=finite and torch.allclose(a,b,rtol=result['rtol'],atol=result['atol'])
            result['probes'].append({'input_shape':list(x.shape),'passed':passed,'finite':finite,
                'max_abs_error':float((a-b).abs().max()),'max_abs_reference_logits':float(a.abs().max()),
                'argmax_agreement':float((a.argmax(1)==b.argmax(1)).float().mean())})
            if not passed:result['status']='not_certified'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['config','checkpoint','report']:p.add_argument('--'+key,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args();torch.set_num_threads(2)
    cfg=yaml.safe_load(Path(args.config).read_text());result=audit(load_model(cfg,args.checkpoint,args.device),seed=cfg['seed'])
    path=Path(args.report);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise SystemExit('Fusion NOT certified; keep unfused model')
