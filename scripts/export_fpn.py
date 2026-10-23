"""Strict checkpoint export of the complete, unfused, single-head Fuse graph."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
import yaml
from models import get_backbone


def load_model(cfg, checkpoint, device='cpu'):
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    expected = cfg['backbone'].get('fusion')
    if 'state_dict' not in payload or payload.get('fusion') != expected:
        raise ValueError('Checkpoint Fuse configuration mismatch or missing checkpoint envelope')
    state = {k[len('network.'):]:v for k,v in payload['state_dict'].items() if k.startswith('network.')}
    model = get_backbone(cfg)
    model.load_state_dict(state,strict=True)
    return model.to(device).eval()


def export(cfg, checkpoint, output_dir, device='cpu', sizes=((768,768),(1024,1024),(33,35))):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Export directory must be empty')
    trained = load_model(cfg,checkpoint,device)
    raw = trained.state_dict()
    reloaded = get_backbone(cfg).to(device).eval();reloaded.load_state_dict(raw,strict=True)
    report = {'graph':'single_head_unfused_eval','fusion_preserved':deepcopy(cfg['backbone'].get('fusion')),
              'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'trainable_params':sum(p.numel() for p in trained.parameters()),'parity':[],
              'structural_reparameterization':'not_requested'}
    generator = torch.Generator().manual_seed(cfg['seed'])
    with torch.inference_mode():
        for h,w in sizes:
            x=torch.randn(1,3,h,w,generator=generator).to(device)
            expected=trained(x);actual=reloaded(x)
            if expected.shape != (1,3,h,w) or not torch.isfinite(actual).all():
                raise ValueError('Invalid export logits')
            error=float((expected-actual).abs().max())
            if error != 0:raise RuntimeError('Export reload changes logits')
            report['parity'].append({'shape':list(x.shape),'max_abs_error':error})
    output.mkdir(parents=True,exist_ok=True)
    torch.save(raw,output/'main_state_dict.pt')
    (output/'main_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['config','checkpoint','output-dir']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    print(json.dumps(export(yaml.safe_load(Path(args.config).read_text()),args.checkpoint,args.output_dir,args.device),indent=2))
