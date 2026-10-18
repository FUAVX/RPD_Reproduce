"""Audit full attached masks before campaign; leave P2 metric semantics unchanged."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image


def audit_dataset(root):
    root=Path(root);result={'dataset_path':str(root.resolve()),'status':'passed','splits':{},'ignore255_policy':'pending team decision if present; legacy val loss/metrics are not masked'}
    for split in ['train','val']:
        images={p.name for p in (root/split/'images').iterdir() if p.suffix.lower() in ('.png','.jpg')}
        masks={p.name for p in (root/split/'semantics').iterdir() if p.suffix.lower() in ('.png','.jpg')}
        if not images or images!=masks:raise ValueError(f'{split}: missing/mismatched image and mask filenames')
        counts=Counter()
        for name in sorted(masks):
            with Image.open(root/split/'semantics'/name) as image: raw=np.array(image)
            if raw.ndim>2:raw=raw[:,:,0]
            ids,n=np.unique(raw,return_counts=True);counts.update({int(k):int(v) for k,v in zip(ids,n)})
        unknown=set(counts)-{0,1,2,3,4,255}
        result['splits'][split]={'images':len(images),'raw_label_pixels':dict(counts),'unique_semantic_labels':sorted(counts),'contains_255':bool(counts[255]),'unknown_labels':sorted(unknown)}
        if unknown:result['status']='blocked_unknown_labels'
        elif counts[255] and result['status']=='passed':result['status']='blocked_pending_ignore255_decision'
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit_dataset(args.dataset_dir)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if result['status']!='passed':raise RuntimeError('Dataset preflight stopped: '+result['status']+'. See DeepSup pending decisions; no legacy metric changes applied.')
