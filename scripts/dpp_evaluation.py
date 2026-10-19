"""Artifact-backed evaluation helpers for unified_model_evaluation_pipeline.ipynb.

All scores are percentages. Historical training metrics and paper references are
kept separate from fresh full-validation inference. No training weights are changed.
"""
from __future__ import annotations
import copy
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn

SELECTIONS = {'best_mIoU': 'best_mIoU_', 'best_weedIoU': 'best_weedIoU_'}
QUARTILES = {'Q1 (Micro)': (0, 131), 'Q2 (Small-Medium)': (131, 321),
             'Q3 (Medium-Large)': (321, 806), 'Q4 (Large)': (806, np.inf)}
PALETTE = {0:(35,35,35), 1:(46,204,113), 2:(231,76,60), 255:(255,255,255)}
STANDARD_GALLERY_IMAGES = [
    '05-15_00173_P0030792.png','05-15_00074_P0030943.png',
    '05-26_00159_P0034016.png','05-26_00159_P0034121.png',
    '06-05_00094_P0037822.png','06-05_00114_P0037991.png',
]
SOURCE_FILES = [
    'models/__init__.py','models/rpdnet/RepDWNet.py','models/rpdnet/rep_dw.py',
    'models/rpdnet/RPDNet.py','models/rpdnet/RPD_Module.py','models/rpdnet/RPD_ops.py',
    'models/rpdnet/dpp.py','models/rpdnet/downsampling.py',
    'datasets/augmentations_normalize.py','datasets/pdc_common.py',
]


def evaluation_signature(workspace,entry,dataset_hash,device,mode):
    device=torch.device(device)
    source_files=SOURCE_FILES+(['models/rpdnet/aaa.py'] if entry.get('family')=='AAA' else [])
    return {'checkpoint_fingerprint':entry['fingerprint'],'dataset_manifest':dataset_hash,
            'mode':mode,'device':str(device),'torch':torch.__version__,
            'gpu_name':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'backbone_config_sha256':hashlib.sha256(json.dumps(entry['config']['backbone'],sort_keys=True).encode()).hexdigest(),
            'source_hashes':{name:hashlib.sha256((Path(workspace)/'RPD_P2'/name).read_bytes()).hexdigest() for name in source_files},
            'evaluation_protocol':'native-772-plantids-globalcm-unfused-profiler-v2'}


def workspace_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for p in [start, *start.parents]:
        if (p/'model').is_dir() and (p/'RPD_P2').is_dir(): return p
        if (p/'workspace/model').is_dir(): return p/'workspace'
    raise FileNotFoundError('Cannot locate workspace/model and workspace/RPD_P2')


def load_source(workspace):
    root = (Path(workspace)/'RPD_P2').resolve()
    if str(root) not in sys.path: sys.path.insert(0, str(root))
    for package in ['models','datasets']:
        loaded=sys.modules.get(package)
        if loaded is not None and not str(getattr(loaded,'__file__','')).startswith(str(root)):
            raise RuntimeError(f'{package} was imported from another source; restart the notebook kernel before evaluation')
    from models import get_backbone
    from models.rpdnet.RPD_Module import RPD_model_deploy
    from models.rpdnet.dpp import LiteDPPSym
    from datasets.augmentations_normalize import SingleImageNormalizer
    return get_backbone, RPD_model_deploy, LiteDPPSym, SingleImageNormalizer


class Artifact:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists(): raise FileNotFoundError(self.path)
        self.names = zipfile.ZipFile(self.path).namelist() if self.path.suffix == '.zip' else [str(p.relative_to(self.path)) for p in self.path.rglob('*') if p.is_file()]

    def match(self, suffix):
        matches = [n for n in self.names if n.endswith(suffix)]
        if len(matches) != 1: raise ValueError(f'{self.path.name}: expected one {suffix}, found {matches}')
        return matches[0]

    def read(self, name):
        if self.path.suffix == '.zip':
            with zipfile.ZipFile(self.path) as z: return z.read(name)
        return (self.path/name).read_bytes()

    def json(self, suffix): return json.loads(self.read(self.match(suffix)))

    def checkpoint(self, selection):
        matches = [n for n in self.names if '/checkpoints/' in '/'+n and SELECTIONS[selection] in Path(n).name and n.endswith('.ckpt')]
        if len(matches) != 1: raise ValueError(f'Ambiguous/missing checkpoint for {selection}: {matches}')
        return matches[0]

    def config(self):
        import yaml
        candidates = [n for n in self.names if n.endswith('runtime_config.yaml')]
        if not candidates: candidates = [n for n in self.names if Path(n).name.startswith('active_config_') and n.endswith('.yaml')]
        if len(candidates) != 1: raise ValueError('No unique effective/active config')
        return yaml.safe_load(self.read(candidates[0]))

    def history(self): return pd.read_csv(io.BytesIO(self.read(self.match('training_output/val_history.csv'))))


def entry_training_cost(entry):
    if not entry['stages']:
        return {}
    suffix='training_output/training_cost.json' if entry.get('family')=='AAA' else 'training_output/dpp_parameters/training_cost.json'
    return Artifact(entry['artifact']).json(suffix)


def discover_artifacts(workspace):
    workspace = Path(workspace)
    wdice = workspace/'model/Loss_Models/pre-model/P2_Ex7_RepDWNet_WDice-Loss_artifacts.zip'
    if not wdice.exists(): wdice = wdice.with_suffix('')
    entries = [{'name':'RepDWNet+WDice', 'artifact':str(wdice), 'stages':[]}]
    for ex, placement, stages in [(1,'Early',[1,2]), (2,'Late',[3,4]), (3,'All',[1,2,3,4])]:
        candidates = list((workspace/'model/Down_Models/DPP').glob(f'P3_DPP_Ex{ex}_{placement}_*_artifacts.zip'))
        if len(candidates) != 1: raise ValueError(f'Expected exactly one DPP {placement} artifact, found {candidates}')
        entries.append({'name':f'DPP-{placement}', 'artifact':str(candidates[0]), 'stages':stages})
    return entries


def checkpoint_payload(artifact, member):
    payload = torch.load(io.BytesIO(artifact.read(member)), map_location='cpu', weights_only=False)
    raw = payload.get('state_dict', payload)
    if any(k.startswith('network.') for k in raw): state = {k[len('network.'):]:v for k,v in raw.items() if k.startswith('network.')}
    elif all(k.startswith('model.') for k in raw): state = {k[len('model.'):]:v for k,v in raw.items()}
    else: state = raw
    if not state or not all(isinstance(v,torch.Tensor) for v in state.values()): raise ValueError('Invalid backbone state_dict')
    digest = hashlib.sha256()
    for k,v in sorted(state.items()):
        digest.update((k+str(v.dtype)+str(tuple(v.shape))).encode()); digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return payload, state, digest.hexdigest()


def metric_values(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float); fp = cm.sum(0)-tp; fn = cm.sum(1)-tp
    def divide(a,b): return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=np.asarray(b)!=0)
    iou = divide(tp,tp+fp+fn); precision = divide(tp,tp+fp); recall = divide(tp,tp+fn)
    f1 = divide(2*tp,2*tp+fp+fn)
    crop_pixels = int(cm[1].sum()); unsafe = int(cm[1,2])
    return dict(mIoU=float(iou.mean()*100), soil_iou=float(iou[0]*100), crop_iou=float(iou[1]*100),
                weed_iou=float(iou[2]*100), weed_prec=float(precision[2]*100), weed_rec=float(recall[2]*100),
                weed_f1=float(f1[2]*100), crop_safety_err_pct=float(unsafe/crop_pixels*100) if crop_pixels else None,
                unsafe_crop_pixels=unsafe)


def confusion_matrix(gt,pred):
    valid = np.isin(gt,[0,1,2])
    if gt.shape != pred.shape or not np.isin(pred,[0,1,2]).all(): raise ValueError('Shape or predicted labels invalid')
    return np.bincount(3*gt[valid].astype(np.int64)+pred[valid].astype(np.int64),minlength=9).reshape(3,3)


class SemanticMetricsAccumulator:
    def __init__(self,num_classes=3):
        if num_classes != 3: raise ValueError('PhenoBench evaluator expects 3 classes')
        self.total_cm = np.zeros((3,3),dtype=np.int64); self.per_image_records=[]
    def update(self,gt,pred,image_name='',subset=''):
        cm = confusion_matrix(gt,pred); self.total_cm += cm
        row = {'image':image_name,'subset':subset,**metric_values(cm),'cm':cm.tolist()}
        self.per_image_records.append(row); return row
    def compute_global_metrics(self): return metric_values(self.total_cm)
    @staticmethod
    def calculate_metrics(cm,crop_px=None,unsafe_px=None): return metric_values(cm)


def lighting_summary(records):
    rows=[]
    for subset in ['Sunny I','Sunny II','Overcast','Unknown']:
        group=[r for r in records if r['subset']==subset]
        if group: rows.append({'subset':subset,'images':len(group),**metric_values(sum((np.array(r['cm']) for r in group),np.zeros((3,3),dtype=np.int64)))})
    return pd.DataFrame(rows)


def remap_gt(raw):
    raw=np.asarray(raw).squeeze()
    if not np.isin(raw,[0,1,2,3,4,255]).all(): raise ValueError('Unknown semantic label; refusing to map silently to soil')
    gt=raw.copy();gt[raw==3]=1;gt[raw==4]=2;return gt.astype(np.uint8)


def instance_statistics(gt,pred,instances):
    """GT plant-ID areas; other-size weeds excluded from each bin's negatives.

    Non-weed false positives are shared across bins, not additively attributed.
    """
    weed=gt==2; predicted=pred==2; valid=np.isin(gt,[0,1,2])
    ids=instances[weed].astype(np.int64)
    if np.any(ids==0): raise ValueError('Weed pixels with instance ID0; size definition needs an explicit policy')
    areas=np.bincount(ids,minlength=1); recovered=np.bincount(ids,weights=predicted[weed],minlength=len(areas))
    fp=int((predicted & valid & ~weed).sum()); rows={}
    for name,(lo,hi) in QUARTILES.items():
        chosen=(areas>lo)&(areas<=hi); chosen[0]=False
        count=int(chosen.sum()); tp=int(recovered[chosen].sum());total=int(areas[chosen].sum());fn=total-tp
        hits=int((recovered[chosen]/areas[chosen]>=.5).sum())
        rows[name]={'instances':count,'gt_pixels':total,'tp':tp,'fp_shared_nonweed':fp,'fn':fn,'hits_05':hits}
    return rows


def size_metrics(counts):
    n=counts['instances'];tp=counts['tp'];fp=counts['fp_shared_nonweed'];fn=counts['fn']
    def ratio(a,b):return 100*a/b if b else None
    return {**counts,'weed_iou':ratio(tp,tp+fp+fn),'weed_prec':ratio(tp,tp+fp),
            'weed_rec':ratio(tp,tp+fn),'weed_f1':ratio(2*tp,2*tp+fp+fn),
            'hr_05':ratio(counts['hits_05'],n)}


def failure_proxies(gt,pred,instances):
    valid=np.isin(gt,[0,1,2]);boundary=np.zeros_like(valid)
    horizontal=(gt[:,:-1]!=gt[:,1:])&valid[:,:-1]&valid[:,1:]
    vertical=(gt[:-1]!=gt[1:])&valid[:-1]&valid[1:]
    boundary[:,:-1]|=horizontal;boundary[:,1:]|=horizontal
    boundary[:-1]|=vertical;boundary[1:]|=vertical
    micro=instance_statistics(gt,pred,instances)['Q1 (Micro)']
    ids=instances[gt==2].astype(np.int64);areas=np.bincount(ids,minlength=1);recovered=np.bincount(ids,weights=(pred==2)[gt==2],minlength=len(areas))
    selected=(areas>0)&(areas<=131);selected[0]=False
    return {'boundary_error_pixels':int((boundary&valid&(gt!=pred)).sum()),
            'soil_false_weed_pixels':int(((gt==0)&(pred==2)).sum()),
            'micro_instances_missed_completely':int((selected&(recovered==0)).sum()),
            'micro_instances':micro['instances'],
            'interior_crop_weed_confusion_pixels':int((~boundary&valid&(((gt==1)&(pred==2))|((gt==2)&(pred==1)))).sum())}


def recipe_audit(prepared):
    control=next(e['config'] for e in prepared if e['name']=='RepDWNet+WDice')
    rows=[]
    for entry in prepared:
        a=copy.deepcopy(entry['config']);b=copy.deepcopy(control)
        a['experiment']=b['experiment']
        a['backbone'].pop('downsampling',None);b['backbone'].pop('downsampling',None)
        # These are runtime paths/device overrides, not model/loss/schedule differences.
        overrides={}
        for section,key in [('data','path_to_dataset'),('data','num_workers'),('train','devices'),('val','devices')]:
            overrides[section+'.'+key]={'candidate':a[section].get(key),'control':b[section].get(key)}
            a[section][key]=b[section].get(key)
        if a!=b:raise ValueError(f"Unexpected recipe drift beyond DS/metadata/runtime overrides: {entry['name']}")
        rows.append({'model':entry['name'],'selection':entry['selection'],'recipe_matches_WDice':True,'runtime_overrides':json.dumps(overrides)})
    return pd.DataFrame(rows)


def prepare_entries(workspace,selections=('best_mIoU','best_weedIoU')):
    prepared=[]
    for entry in discover_artifacts(workspace):
        artifact=Artifact(entry['artifact']);cfg=artifact.config()
        actual=cfg['backbone'].get('downsampling',{}).get('stages',[])
        if actual != entry['stages']: raise ValueError('Placement/config mismatch')
        for selection in selections:
            member=artifact.checkpoint(selection);payload,state,fingerprint=checkpoint_payload(artifact,member)
            prepared.append({**entry,'selection':selection,'member':member,'epoch':int(payload['epoch']),
                             'fingerprint':fingerprint,'config':cfg})
    return prepared


def dpp_parameter_analysis(workspace,prepared,output):
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    raw_rows=[];stats=[];trajectory=[];validation=[]
    for entry in prepared:
        if not entry['stages']:continue
        artifact=Artifact(entry['artifact']);_,state,_=checkpoint_payload(artifact,entry['member'])
        event='checkpoint_val_mIoU.json' if entry['selection']=='best_mIoU' else 'checkpoint_val_weed_iou.json'
        logged=artifact.json('training_output/dpp_parameters/'+event)
        if logged['epoch']!=entry['epoch']: raise ValueError('DPP export checkpoint epoch mismatch')
        for key,value in state.items():
            if not key.endswith('log_alpha'):continue
            layer=key.removesuffix('.log_alpha');alpha=value.clamp(max=10).exp().numpy();lam=state[layer+'.log_lambda'].exp().numpy()
            np.testing.assert_allclose(alpha,logged['layers'][layer]['alpha']['values'],rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(lam,logged['layers'][layer]['lambda']['values'],rtol=1e-6,atol=1e-7)
            validation.append({'model':entry['name'],'selection':entry['selection'],'layer':layer,'checkpoint_export_matches':True})
            for ch,(a,l) in enumerate(zip(alpha,lam)):raw_rows.append({'model':entry['name'],'selection':entry['selection'],'epoch':entry['epoch'],'event':'selected_checkpoint','layer':layer,'channel':ch,'alpha':float(a),'lambda':float(l)})
        if entry['selection'] != 'best_mIoU':continue
        for event in ['initial','final']:
            log=artifact.json('training_output/dpp_parameters/'+event+'.json')
            for layer,params in log['layers'].items():
                for ch,(a,l) in enumerate(zip(params['alpha']['values'],params['lambda']['values'])):
                    raw_rows.append({'model':entry['name'],'selection':'training_trace','epoch':log['epoch'],'event':event,'layer':layer,'channel':ch,'alpha':a,'lambda':l})
        history=artifact.read(artifact.match('training_output/dpp_parameters/history.jsonl')).decode()
        for line in history.splitlines():
            log=json.loads(line)
            if log['event'] not in ['initial','epoch_end','final']:continue
            for layer,params in log['layers'].items():
                trajectory.append({'model':entry['name'],'layer':layer,'event':log['event'],'epoch':log['epoch'],
                                   'alpha_mean':params['alpha']['mean'],'lambda_mean':params['lambda']['mean']})
    raw=pd.DataFrame(raw_rows)
    for keys,frame in raw.groupby(['model','selection','event','layer'],sort=False):
        row=dict(zip(['model','selection','event','layer'],keys));row['channels']=len(frame)
        for metric in ['alpha','lambda']:
            v=frame[metric].to_numpy();row.update({metric+'_'+k:float(x) for k,x in [('mean',v.mean()),('median',np.median(v)),('std',v.std()),('min',v.min()),('max',v.max())]})
        stats.append(row)
    raw.to_csv(output/'dpp_channels.csv',index=False);pd.DataFrame(stats).to_csv(output/'dpp_parameter_summary.csv',index=False)
    pd.DataFrame(trajectory).to_csv(output/'dpp_training_trajectory.csv',index=False)
    pd.DataFrame(validation).to_csv(output/'checkpoint_export_consistency.csv',index=False)
    for metric in ['alpha','lambda']:
        selected=raw[(raw.event=='selected_checkpoint')&(raw.selection=='best_mIoU')]
        groups=list(selected.groupby(['model','layer'],sort=False));fig,axes=plt.subplots(1,2,figsize=(16,5))
        for (model,layer),frame in groups:axes[0].hist(frame[metric],bins='auto',alpha=.4,label=f'{model}/{layer}')
        axes[0].set(xlabel=metric,ylabel='Channels',title='Selected best-mIoU checkpoint');axes[0].legend(fontsize=7)
        tr=pd.DataFrame(trajectory)
        for (model,layer),frame in tr.groupby(['model','layer'],sort=False):axes[1].plot(frame.epoch,frame[metric+'_mean'],label=f'{model}/{layer}')
        axes[1].set(xlabel='Epoch (0-based)',ylabel=metric+' mean',title='Training trajectory');axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(output/f'dpp_{metric}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,6))
    for (model,layer),frame in raw[(raw.event=='selected_checkpoint')&(raw.selection=='best_mIoU')].groupby(['model','layer'],sort=False):ax.scatter(frame.alpha,frame['lambda'],s=16,alpha=.7,label=f'{model}/{layer}')
    ax.set(xlabel='Effective alpha',ylabel='Effective lambda',title='Joint alpha/lambda — checkpoint parameters, not measured pooling selectivity');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(output/'dpp_alpha_lambda_joint.png',dpi=160);plt.close(fig)
    return pd.DataFrame(stats)


def artifact_comparison(workspace,prepared,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=[]
    for entry in prepared:
        artifact=Artifact(entry['artifact']);h=artifact.history();matched=h[h.epoch==entry['epoch']]
        if len(matched)!=1:raise ValueError(f'Checkpoint/history epoch not unique: {entry}')
        row=matched.iloc[0];r={'model':entry['name'],'selection':entry['selection'],'checkpoint_epoch_0based':entry['epoch'],
                             'scope':'historical_training_validation','source_artifact':entry['artifact'],'checkpoint_member':entry['member'],'fingerprint':entry['fingerprint']}
        for k in ['mIoU','soil_iou','crop_iou','weed_iou','weed_precision','weed_recall','weed_f1']:r[k]=float(row[k])*100
        r['trainable_params']=189827+(0 if entry.get('family')=='AAA' else sum(2*(16*2**(s-1)) for s in entry['stages']))
        if entry['stages']:r['observed_fit_hours']=entry_training_cost(entry)['total_fit_seconds']/3600
        else:r['observed_fit_hours']=None
        rows.append(r)
    recipe_audit(prepared).to_csv(output/'recipe_audit.csv',index=False)
    frame=pd.DataFrame(rows)
    for selection in frame.selection.unique():
        group=frame[frame.selection==selection];control=group[group.model=='RepDWNet+WDice'].iloc[0]
        idx=frame.selection==selection
        for metric in ['mIoU','weed_iou','crop_iou','weed_precision','weed_recall','weed_f1']:frame.loc[idx,metric+'_delta_pp_vs_WDice']=frame.loc[idx,metric]-control[metric]
    frame.to_csv(output/'historical_checkpoint_comparison.csv',index=False)
    # Same paper values/table scopes as analyse_p2_loss_experiments.ipynb; no fictional re-inference.
    paper=pd.DataFrame([{'model':'Paper RPD-Net theta=0 [VVV]','table':'II','mIoU':87.92,'soil_iou':99.38,'crop_iou':95.48,'weed_iou':68.90,'params_M':.189},
                        {'model':'Paper RPD-Net theta=0.5','table':'II','mIoU':88.89,'soil_iou':99.37,'crop_iou':95.59,'weed_iou':71.70,'params_M':.189},
                        {'model':'Paper RPD-Net theta=0.5 (mean)','table':'V/VI','mIoU':88.26,'weed_iou':70.03,'weed_precision':73.74,'weed_recall':91.53,'weed_f1':81.68}])
    paper['scope']='paper_reference_only_different_training_protocol';paper.to_csv(output/'paper_references.csv',index=False)
    import matplotlib.pyplot as plt
    for selection in frame.selection.unique():
        g=frame[frame.selection==selection]
        plot_comparison(g,paper[paper.table=='II'],output/f'historical_{selection}.png',selection+' — stored validation, not fresh 772-image inference')
    return frame,paper


def plot_comparison(frame,paper,path,title):
    import matplotlib.pyplot as plt
    combined=pd.concat([frame,paper],ignore_index=True)
    x=np.arange(len(combined));fig,ax=plt.subplots(figsize=(15,6));metrics=['mIoU','weed_iou','crop_iou','soil_iou'];width=.19
    control=frame[frame.model=='RepDWNet+WDice'].iloc[0]
    for i,metric in enumerate(metrics):
        bars=ax.bar(x+(i-1.5)*width,combined[metric],width,label=metric)
        ax.bar_label(bars,fmt='%.2f',fontsize=7,rotation=90,padding=2)
        ax.axhline(control[metric],color=f'C{i}',linestyle='--',linewidth=.9,alpha=.7)
    ax.set_xticks(x,combined.model,rotation=15,ha='right');ax.set(ylabel='Percent',ylim=(50,104),title=title)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.16))
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def colorize_mask(mask):
    rgb=np.zeros((*mask.shape,3),dtype=np.uint8)
    for k,v in PALETTE.items():rgb[mask==k]=v
    return Image.fromarray(rgb)


def render_5column_qualitative_strip(rgb_img,gt_mask,pred_base,pred_var,meta,zoom_size=None):
    valid=np.isin(gt_mask,[0,1,2]);error=np.zeros((*gt_mask.shape,3),dtype=np.uint8);error[valid&(pred_var!=gt_mask)]=(255,0,255)
    panels=[rgb_img.convert('RGB'),colorize_mask(gt_mask),colorize_mask(pred_base),colorize_mask(pred_var),Image.fromarray(error)]
    if zoom_size:
        changed=np.argwhere(valid&(pred_base!=pred_var)&((gt_mask==2)|(pred_base==2)|(pred_var==2)))
        y,x=np.median(changed,axis=0).astype(int) if len(changed) else np.array(gt_mask.shape)//2
        half=zoom_size//2;left=max(0,min(int(x)-half,gt_mask.shape[1]-zoom_size));top=max(0,min(int(y)-half,gt_mask.shape[0]-zoom_size))
        panels=[p.crop((left,top,left+zoom_size,top+zoom_size)) for p in panels]
    width=400;header=65;gap=8;strip=Image.new('RGB',(5*width+4*gap,width+header),(20,20,20));draw=ImageDraw.Draw(strip)
    names=[meta.get('image','RGB'),'Ground truth',meta.get('base_name','WDice'),meta.get('var_name','DPP'),'Pixel error: magenta']
    for i,(panel,name) in enumerate(zip(panels,names)):
        strip.paste(panel.resize((width,width),Image.Resampling.LANCZOS if i==0 else Image.Resampling.NEAREST),(i*(width+gap),header));draw.text((i*(width+gap)+5,8),name,fill='white')
    return strip


def benchmark_forward_latency(model,device,resolution=(768,768),warmup_runs=30,timed_runs=200):
    device=torch.device(device)
    if device.type!='cuda':return {'status':'not_measured_no_CUDA','scope':'GPU-only protocol; no CPU numbers substituted','warmup_runs':warmup_runs,'timed_runs':timed_runs,'resolution':list(resolution)}
    model=model.to(device).eval();x=torch.randn(1,3,*resolution,device=device);times=[]
    with torch.inference_mode():
        for _ in range(warmup_runs):model(x)
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        for _ in range(timed_runs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();model(x);end.record();torch.cuda.synchronize(device);times.append(start.elapsed_time(end))
    return {'status':'measured','mean_latency_ms':float(np.mean(times)),'p50_latency_ms':float(np.percentile(times,50)),
            'p95_latency_ms':float(np.percentile(times,95)),'throughput_fps':1000/float(np.mean(times)),
            'peak_allocated_MiB':torch.cuda.max_memory_allocated(device)/2**20,
            'warmup_runs':warmup_runs,'timed_runs':timed_runs,'resolution':list(resolution),'scope':'forward-only batch1 FP32 deployed graph; excludes image IO/preprocessing'}



def benchmark_end_to_end(model,device,image_path,normalizer,warmup_runs=30,timed_runs=200):
    """Warm-cache file decode -> normalize -> transfer -> forward -> argmax/CPU."""
    from torchvision.transforms.functional import to_tensor
    device=torch.device(device)
    if device.type!='cuda':
        return {'status':'not_measured_no_CUDA','scope':'GPU end-to-end native1024; no CPU substitute'}
    def step():
        with Image.open(image_path) as image:
            x=normalizer.normalize(to_tensor(image.convert('RGB')))[None].to(device)
        return model(x).argmax(1).cpu().numpy()
    times=[]
    with torch.inference_mode():
        for _ in range(warmup_runs):step()
        torch.cuda.synchronize(device)
        for _ in range(timed_runs):
            torch.cuda.synchronize(device);start=time.perf_counter();step();torch.cuda.synchronize(device)
            times.append((time.perf_counter()-start)*1000)
    return {'status':'measured','mean_latency_ms':float(np.mean(times)),
            'p50_latency_ms':float(np.percentile(times,50)),'p95_latency_ms':float(np.percentile(times,95)),
            'warmup_runs':warmup_runs,'timed_runs':timed_runs,'image':Path(image_path).name,
            'resolution':[1024,1024],'scope':'single-image warm-cache file decode/normalization/transfer/deployed forward/argmax/CPU; excludes saving predictions'}


def dataset_index(dataset_root,full=True):
    root=Path(dataset_root);paths=sorted((root/'val/images').glob('*.png'))
    if not paths:raise FileNotFoundError(root/'val/images')
    counts={};digest=hashlib.sha256()
    for path in paths:
        subset=lighting_name(path.name);counts[subset]=counts.get(subset,0)+1
        for item in [path,root/'val/semantics'/path.name,root/'val/plant_instances'/path.name]:
            st=item.stat();digest.update(f'{item.resolve()}:{st.st_size}:{st.st_mtime_ns}'.encode())
    if full and (len(paths)!=772 or counts!={'Sunny I':399,'Sunny II':170,'Overcast':203}):
        raise ValueError(f'Expected complete772 validation set and399/170/203 subsets; found{len(paths)}, {counts}')
    return paths,counts,digest.hexdigest()


def lighting_name(filename):
    return {'05-15':'Sunny I','05-26':'Sunny II','06-05':'Overcast'}.get(filename[:5],'Unknown')


def _json_write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def evaluate_checkpoint(workspace,entry,dataset_root,output,device='cpu',mode='full'):
    """Full mode checks native-image train/deploy parity on every evaluated image.

    Calibration mode is explicitly three images (first filename per lighting
    subset). Its scores never enter the full-validation ranking.
    """
    import gc
    from contextlib import nullcontext
    from torch.profiler import profile as torch_profile, ProfilerActivity
    from torchvision.transforms.functional import to_tensor
    get_backbone,convert,DPP,Normalizer=load_source(workspace)
    pool_type=DPP
    if entry.get('family')=='AAA':
        from models.rpdnet.aaa import AAAMaxBlurPool
        pool_type=AAAMaxBlurPool
    device=torch.device(device);root=Path(dataset_root);paths,subset_counts,dataset_hash=dataset_index(root)
    if mode=='calibration':paths=[next(p for p in paths if lighting_name(p.name)==s) for s in ['Sunny I','Sunny II','Overcast']]
    elif mode!='full':raise ValueError(mode)
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cache=output/entry['fingerprint']
    cache.mkdir(exist_ok=True);metadata_path=cache/'evaluation.json'
    signature=evaluation_signature(workspace,entry,dataset_hash,device,mode)
    if metadata_path.exists():
        cached=json.loads(metadata_path.read_text())
        if cached.get('signature')==signature and cached.get('status')=='completed':
            print('[CACHE]',entry['name'],entry['selection'],flush=True);return cached
    artifact=Artifact(entry['artifact']);payload,state,fingerprint=checkpoint_payload(artifact,entry['member'])
    assert fingerprint==entry['fingerprint']
    model=get_backbone(entry['config']).eval();model.load_state_dict(state,strict=True)
    deploy=convert(model,do_copy=True).eval()
    if sum(isinstance(m,pool_type) for m in deploy.modules())!=len(entry['stages']):raise AssertionError('Deployment lost selected downsampling stages')
    profile={'trainable_params_raw':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'deploy_params_raw':sum(p.numel() for p in deploy.parameters()),
             'dpp_learnable_params':sum(p.numel() for m in model.modules() if isinstance(m,DPP) for p in m.parameters())}
    if entry.get('family')=='AAA':
        profile['aaa_buffer_elements']=sum(m.filt.numel() for m in model.modules() if isinstance(m,pool_type))
        profile['aaa_learnable_params']=sum(p.numel() for m in model.modules() if isinstance(m,pool_type) for p in m.parameters())
    model.to(device);deploy.to(device);normalizer=Normalizer();acc=SemanticMetricsAccumulator();train_acc=SemanticMetricsAccumulator()
    size_counts={q:{'instances':0,'gt_pixels':0,'tp':0,'fp_shared_nonweed':0,'fn':0,'hits_05':0} for q in QUARTILES}
    parity_rows=[];latencies=[];start_time=time.perf_counter();max_error=0.;mismatches=0
    (cache/'predictions').mkdir(exist_ok=True)
    for i,path in enumerate(paths):
        rgb=Image.open(path).convert('RGB');gt=remap_gt(np.array(Image.open(root/'val/semantics'/path.name)))
        instances=np.array(Image.open(root/'val/plant_instances'/path.name))
        if gt.shape!=(1024,1024) or rgb.size!=(1024,1024):raise ValueError('Native1024 evaluation cannot resize input/mask')
        x=normalizer.normalize(to_tensor(rgb)).unsqueeze(0).to(device)
        # Count executed aten::conv2d, including functional convolutions in RPD.
        # nn.Conv2d hooks alone miss these operations in the unfused graph.
        with torch.inference_mode():
            with torch_profile(activities=[ProfilerActivity.CPU],with_flops=True) if i==0 else nullcontext() as pa:
                a=model(x)
            with torch_profile(activities=[ProfilerActivity.CPU],with_flops=True) if i==0 else nullcontext() as pb:
                b=deploy(x)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite model outputs')
            delta=float((a-b).abs().max());train_pred=a.argmax(1)[0].cpu().numpy().astype(np.uint8);pred=b.argmax(1)[0].cpu().numpy().astype(np.uint8)
        if i==0:
            fa=sum(e.flops for e in pa.key_averages() if e.key=='aten::conv2d')
            fb=sum(e.flops for e in pb.key_averages() if e.key=='aten::conv2d')
            if not fa or not fb:raise RuntimeError('Profiler did not count convolutions; refusing to emit zero MACs')
            profile.update(train_Conv_MACs_1024=fa//2,deploy_Conv_MACs_1024=fb//2,deploy_Conv_FLOPs_1024=fb,
                           MAC_scope='Executed aten::conv2d only, including functional convolutions; excludes DPP non-Conv operations, normalization and interpolation')
            del pa,pb
        mismatch=int((train_pred!=pred).sum());max_error=max(max_error,delta);mismatches+=mismatch
        parity_rows.append({'image':path.name,'max_abs_logits':delta,'mismatched_argmax_pixels':mismatch})
        # Original trained eval graph is authoritative; never silently rely on failed fusion.
        row=acc.update(gt,train_pred,path.name,lighting_name(path.name));train_acc.update(gt,pred,path.name,lighting_name(path.name))
        pred=train_pred
        row.update(failure_proxies(gt,pred,instances));row['weed_gt_pixels']=int((gt==2).sum());row['weed_union_pixels']=int(((gt==2)|(pred==2)).sum())
        sizes=instance_statistics(gt,pred,instances)
        for q,values in sizes.items():
            for k,v in values.items():size_counts[q][k]+=v
        Image.fromarray(pred).save(cache/'predictions'/path.name)
        if (i+1)%25==0 or i+1==len(paths):print(f"[EVAL] {entry['name']} {entry['selection']} {i+1}/{len(paths)}, max_delta={max_error:.3g}, elapsed={time.perf_counter()-start_time:.1f}s",flush=True)
        del a,b,x
    pd.DataFrame([{**r,'cm':json.dumps(r['cm'])} for r in acc.per_image_records]).to_csv(cache/'per_image.csv',index=False)
    lighting_summary(acc.per_image_records).to_csv(cache/'lighting.csv',index=False)
    pd.DataFrame([{'size_bin':q,**size_metrics(c)} for q,c in size_counts.items()]).to_csv(cache/'weed_size.csv',index=False)
    pd.DataFrame(parity_rows).to_csv(cache/'parity_per_image.csv',index=False)
    # Keep only the deployment graph resident for its memory/latency benchmark.
    if device.type=='cuda':
        model.cpu();torch.cuda.empty_cache()
    benchmark=benchmark_forward_latency(deploy,device)
    result={'status':'completed','signature':signature,'mode':mode,'image_count':len(paths),'subset_counts_full_dataset':subset_counts,
            'model_name':entry['name'],'checkpoint_member':entry['member'],'epoch_0based':entry['epoch'],'cache_dir':str(cache.resolve()),
            'metrics':acc.compute_global_metrics(),'global_confusion_matrix':acc.total_cm.tolist(),
            'metrics_graph':'unfused_eval', 'deployed_eval_metrics':train_acc.compute_global_metrics(),
            'parity':{'passed':bool(max_error<1e-4),'tolerance':1e-4,'max_abs_logits':max_error,'mismatched_argmax_pixels':mismatches,'image_count':len(paths),'scope':'all images in this evaluation mode'},
            'profile':profile,'latency':benchmark,'end_to_end_latency':benchmark_end_to_end(deploy,device,paths[0],normalizer),'elapsed_seconds':time.perf_counter()-start_time,
            'size_definition':'GT plant IDs intersected with remapped Weed; train-derived131/321/806 thresholds; valid non-Weed FPs reused per bin; other-size GT weeds excluded',
            'normalization':'actual RPD_P2 SingleImageNormalizer, unchanged from training'}
    if entry.get('family')=='AAA':
        result['latency_native1024']=benchmark_forward_latency(deploy,device,resolution=(1024,1024))
        result['elapsed_seconds']=time.perf_counter()-start_time
    _json_write(metadata_path,result)
    del model,deploy;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def export_qualitative(workspace,dataset_root,selection,entries,results,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    control=next(e for e in entries if e['name']=='RepDWNet+WDice' and e['selection']==selection)
    base=results[control['fingerprint']];base_cache=Path(base['cache_dir']);base_rows=pd.read_csv(base_cache/'per_image.csv').set_index('image');manifest=[]
    for entry in entries:
        if entry['selection']!=selection or entry['name']=='RepDWNet+WDice':continue
        cache=Path(results[entry['fingerprint']]['cache_dir']);rows=pd.read_csv(cache/'per_image.csv').set_index('image')
        matched=rows.join(base_rows,lsuffix='_variant',rsuffix='_base',how='inner')
        matched=matched[(matched.weed_gt_pixels_base>0)]
        matched['delta_weed_iou']=matched.weed_iou_variant-matched.weed_iou_base
        matched['delta_crop_safety_error']=matched.crop_safety_err_pct_variant-matched.crop_safety_err_pct_base
        matched.to_csv(output/f'{entry["name"]}_{selection}_paired_per_image.csv')
        if matched.empty:continue
        positive=matched.delta_weed_iou.idxmax();trade=matched[(matched.delta_crop_safety_error>0)&(matched.delta_weed_iou>0)]
        trade_name=trade.delta_weed_iou.idxmax() if len(trade) else matched.delta_weed_iou.idxmin()
        cases=[('largest_weed_gain' if matched.loc[positive,'delta_weed_iou']>0 else 'no_positive_case_highest_weed_delta',positive),('crop_safety_tradeoff' if len(trade) else 'lowest_weed_delta',trade_name)]
        # Fixed cases from the unified instruction supplement the selected contrasts.
        cases.extend((f'standard_{Path(name).stem}',name) for name in STANDARD_GALLERY_IMAGES if name in matched.index)
        for kind,fname in cases:
            rgb=Image.open(Path(dataset_root)/'val/images'/fname).convert('RGB');gt=remap_gt(np.array(Image.open(Path(dataset_root)/'val/semantics'/fname)))
            a=np.array(Image.open(base_cache/'predictions'/fname));b=np.array(Image.open(cache/'predictions'/fname))
            meta={'image':fname,'base_name':'WDice','var_name':entry['name']}
            for zoom in [None,512]:
                strip=render_5column_qualitative_strip(rgb,gt,a,b,meta,zoom)
                strip.save(output/f'{entry["name"]}_{selection}_{kind}_{"full" if zoom is None else "zoom512"}.png')
            manifest.append({'model':entry['name'],'selection':selection,'case':kind,'image':fname,
                             'delta_weed_iou_pp':float(matched.loc[fname,'delta_weed_iou']),
                             'delta_crop_safety_error_pp':float(matched.loc[fname,'delta_crop_safety_error'])})
    _json_write(output/'selection_manifest.json',manifest)


def export_diagnostic_comparisons(entries,results,output,mode):
    """Combine cached observations without rerunning or selecting new checkpoints."""
    output=Path(output);lighting=[];sizes=[];failures=[];efficiency=[];legacy=[]
    fault_columns=['boundary_error_pixels','soil_false_weed_pixels',
                   'micro_instances_missed_completely','micro_instances',
                   'interior_crop_weed_confusion_pixels']
    for entry in entries:
        result=results[entry['fingerprint']];cache=Path(result['cache_dir'])
        identity={'model':entry['name'],'selection':entry['selection']}
        light=pd.read_csv(cache/'lighting.csv');size=pd.read_csv(cache/'weed_size.csv')
        cm=np.array(result['global_confusion_matrix'])
        if int(size.gt_pixels.sum())!=int(cm[2].sum()):
            raise AssertionError('Size bins do not cover exactly all valid Weed pixels')
        if not (size.fp_shared_nonweed==int(cm[:2,2].sum())).all():
            raise AssertionError('Size-bin shared FP differs from global Weed FP')
        lighting.extend([{**identity,**r} for r in light.to_dict('records')])
        sizes.extend([{**identity,**r} for r in size.to_dict('records')])
        per_image=pd.read_csv(cache/'per_image.csv')
        if mode=='full':
            # Diagnostic reconstruction of the legacy 2-rank batch4 drop_last
            # validation; the authoritative quality result remains all772.
            prefix=per_image.iloc[:768]
            prefix_cm=sum((np.array(json.loads(v),dtype=np.int64) for v in prefix.cm),np.zeros((3,3),dtype=np.int64))
            observed=metric_values(prefix_cm)
            history=Artifact(entry['artifact']).history()
            recorded=history[history.epoch==entry['epoch']].iloc[0]
            row={**identity,'images':768,'diagnostic_scope':'first768 sorted images; legacy2-rank batch4 drop_last reconstruction'}
            for metric in ['mIoU','soil_iou','crop_iou','weed_iou','weed_f1']:
                row[metric]=observed[metric]
                row[metric+'_delta_pp_vs_training_log']=observed[metric]-100*float(recorded[metric])
            legacy.append(row)
        fault={k:int(per_image[k].sum()) for k in fault_columns}
        fault['micro_missed_completely_pct']=100*fault['micro_instances_missed_completely']/fault['micro_instances'] if fault['micro_instances'] else None
        failures.append({**identity,**fault})
        cost=entry_training_cost(entry)
        efficiency.append({**identity,**result['profile'],
            'fit_hours_rank_zero':cost.get('total_fit_seconds',float('nan'))/3600,
            'mean_epoch_seconds_including_validation':cost.get('mean_epoch_wall_seconds'),
            'training_peak_allocated_bytes_rank_zero':cost.get('peak_gpu_allocated_bytes_rank_zero'),
            'training_world_size':cost.get('world_size'),
            'forward_ms':result['latency'].get('mean_latency_ms'),
            'forward_p50_ms':result['latency'].get('p50_latency_ms'),
            'forward_p95_ms':result['latency'].get('p95_latency_ms'),
            'forward_fps':result['latency'].get('throughput_fps'),
            'forward_peak_allocated_MiB':result['latency'].get('peak_allocated_MiB'),
            'end_to_end_native1024_ms':result.get('end_to_end_latency',{}).get('mean_latency_ms')})
    for kind,records,stratum,metrics in [
        ('lighting',lighting,'subset',['mIoU','weed_iou','weed_prec','weed_rec','weed_f1','crop_safety_err_pct']),
        ('weed_size',sizes,'size_bin',['weed_iou','weed_prec','weed_rec','weed_f1','hr_05']),
        ('failure_proxies',failures,None,['micro_missed_completely_pct'])]:
        frame=pd.DataFrame(records)
        keys=['selection']+([stratum] if stratum else [])
        baseline=frame[frame.model=='RepDWNet+WDice'][keys+metrics]
        frame=frame.merge(baseline,on=keys,how='left',suffixes=('','_WDice'),validate='many_to_one')
        for metric in metrics:frame[metric+'_delta_pp_vs_WDice']=frame[metric]-frame.pop(metric+'_WDice')
        frame.to_csv(output/f'{mode}_{kind}_comparison.csv',index=False)
    pd.DataFrame(efficiency).to_csv(output/f'{mode}_efficiency_comparison.csv',index=False)
    if legacy:pd.DataFrame(legacy).to_csv(output/'legacy768_reconstruction.csv',index=False)


def run_master_unified_evaluation(dataset_root,model_entries,output_dir,workspace=None,device=None,mode='full'):
    workspace=Path(workspace or workspace_root());output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    device=torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'));results={};rows=[]
    for entry in model_entries:
        if entry['fingerprint'] not in results:results[entry['fingerprint']]=evaluate_checkpoint(workspace,entry,dataset_root,output/'inference'/mode,device,mode)
        r=results[entry['fingerprint']]
        rows.append({'model':entry['name'],'selection':entry['selection'],'epoch_0based':entry['epoch'],
                     'scope':mode,'images':r['image_count'],'metrics_graph':r['metrics_graph'],**r['metrics'],**r['profile'],
                     'parity_passed':r['parity']['passed'],'parity_max_abs_logits':r['parity']['max_abs_logits'],
                     'latency_ms':r['latency'].get('mean_latency_ms'),'latency_status':r['latency']['status'],
                     'end_to_end_latency_ms':r.get('end_to_end_latency',{}).get('mean_latency_ms'),
                     'checkpoint_fingerprint':entry['fingerprint']})
    frame=pd.DataFrame(rows)
    for selection in frame.selection.unique():
        idx=frame.selection==selection;control=frame[idx&(frame.model=='RepDWNet+WDice')].iloc[0]
        for metric in ['mIoU','weed_iou','crop_iou','weed_prec','weed_rec','weed_f1','crop_safety_err_pct']:
            frame.loc[idx,metric+'_delta_pp_vs_WDice']=frame.loc[idx,metric]-control[metric]
    frame.to_csv(output/f'{mode}_evaluation_summary.csv',index=False)
    metadata={'mode':mode,'dataset':str(Path(dataset_root).resolve()),'device':str(device),'torch':torch.__version__,
              'manifest_written_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'hardware':{'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                          'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
                          'cpu_threads':torch.get_num_threads()},
              'unique_weight_states':len(results),'selected_checkpoints':len(model_entries),
              'full_validation_complete':mode=='full' and all(r['image_count']==772 for r in results.values()),
              'all_deploy_parity_passed':all(r['parity']['passed'] for r in results.values()),
              'protocol_status':'deploy_parity_passed' if all(r['parity']['passed'] for r in results.values()) else 'deploy_parity_failed_unfused_quality_only',
              'paper_rows':'reference only; never inferred without checkpoints','results':results}
    _json_write(output/f'{mode}_evaluation_manifest.json',metadata)
    export_diagnostic_comparisons(model_entries,results,output,mode)
    for selection in frame.selection.unique():
        export_qualitative(workspace,dataset_root,selection,model_entries,results,output/'qualitative'/mode/selection)
    if mode=='full':
        paper=pd.read_csv(output/'artifacts/paper_references.csv')
        for selection in frame.selection.unique():plot_comparison(frame[frame.selection==selection],paper[paper.table=='II'],output/f'full_{selection}.png','Fresh 772-image validation; paper columns reference only')
    return frame,metadata
