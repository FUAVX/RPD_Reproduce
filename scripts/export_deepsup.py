#!/usr/bin/env python3
"""Strictly load the training model, strip its auxiliary head, export main only.

Unfused main export is authoritative. Optional fusion must pass absolute 1e-4
logit parity on all probe shapes; failure never weakens the threshold.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
import yaml
from models import get_backbone
from models.rpdnet.RPD_Module import RPD_model_deploy


def load_training_model(cfg, checkpoint, device='cpu'):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'state_dict' in payload and payload.get('deep_supervision') != cfg['backbone'].get('deep_supervision'):
        raise ValueError('Checkpoint DeepSup configuration mismatch')
    if ('state_dict' in payload and ('downsampling' in payload or cfg['backbone'].get('deep_supervision') is not None)
            and payload.get('downsampling') != cfg['backbone'].get('downsampling')):
        raise ValueError('Checkpoint downsampling configuration mismatch')
    raw = payload.get('state_dict', payload)
    state = {k[len('network.'):]: v for k,v in raw.items() if k.startswith('network.')} if any(k.startswith('network.') for k in raw) else raw
    model = get_backbone(cfg)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def export(cfg, checkpoint, output_dir, device='cpu', fuse=False, sizes=((768,768),(1024,1024),(33,35))):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty export directory to avoid stale weights/metadata')
    output.mkdir(parents=True, exist_ok=True)
    trained = load_training_model(cfg, checkpoint, device)
    main = trained.inference_model()
    assert main.aux_head is None and not any(k.startswith('aux_head.') for k in main.state_dict())
    generator = torch.Generator().manual_seed(cfg['seed'])
    probes = [torch.randn(1,3,*shape,generator=generator).to(device) for shape in sizes]
    report = {'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'graph':'main_only_unfused_eval','auxiliary_removed':True,'stripping_parity':[],
              'main_trainable_params':sum(p.numel() for p in main.parameters()),
              'downsampling_preserved':deepcopy(getattr(main, 'downsampling', None)),
              'fusion':{'status':'not_requested','absolute_logit_tolerance':1e-4}}
    with torch.inference_mode():
        for x in probes:
            error = float((trained(x)-main(x)).abs().max())
            if error != 0:raise RuntimeError('Stripping changed the main graph output')
            report['stripping_parity'].append({'shape':list(x.shape),'max_abs_error':error})
    torch.save(main.cpu().state_dict(), output/'main_state_dict.pt')
    inference_cfg=deepcopy(cfg);inference_cfg['backbone'].pop('deep_supervision',None)
    inference_cfg['backbone']['deploy']=False
    (output/'main_config.yaml').write_text(yaml.safe_dump(inference_cfg,sort_keys=False))
    if fuse:
        main=main.to(device)
        fused=RPD_model_deploy(main,do_copy=True).eval()
        with torch.inference_mode():
            errors=[float((main(x)-fused(x)).abs().max()) for x in probes]
        valid=all(error<=1e-4 for error in errors)
        report['fusion']={'status':'passed' if valid else 'failed','absolute_logit_tolerance':1e-4,'max_abs_errors':errors}
        if valid:
            torch.save(fused.cpu().state_dict(),output/'fused_state_dict.pt')
            fused_cfg=deepcopy(inference_cfg);fused_cfg['backbone']['deploy']=True
            (output/'fused_config.yaml').write_text(yaml.safe_dump(fused_cfg,sort_keys=False))
        elif (output/'fused_state_dict.pt').exists():
            (output/'fused_state_dict.pt').unlink()
            (output/'fused_config.yaml').unlink(missing_ok=True)
    (output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    if fuse and report['fusion']['status']=='failed':
        raise RuntimeError('Fusion parity failed. Unfused main export retained; no fused weights certified.')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--fuse',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    print(json.dumps(export(yaml.safe_load(Path(args.config).read_text()),args.checkpoint,args.output_dir,args.device,args.fuse),indent=2))
