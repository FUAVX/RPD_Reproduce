#!/usr/bin/env python3
"""Export stored best-mIoU validation results and consistently profiled model costs.

No inference accuracy re-evaluation or training is performed. See generated notes
and provenance for missing paper values, graph/cost scope and timing estimates.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dpp_evaluation import Artifact, checkpoint_payload
from models import get_backbone

METRICS = {'mIoU (%)': 'mIoU', 'Soil IoU (%)': 'soil_iou',
           'Crop IoU (%)': 'crop_iou', 'Weed IoU (%)': 'weed_iou',
           'Weed Precision (%)': 'weed_precision', 'Weed Recall (%)': 'weed_recall',
           'Weed F1 (%)': 'weed_f1'}
FIELDS = ['Model', 'Time-Training (h)', 'MACs (G)', 'FLOPs (G)', 'Params (M)', 'Params (count)',
          *METRICS, 'Checkpoint epoch (0-based)', 'Metric source', 'Time source']


def training_time(artifact):
    for member in ['training_output/training_cost.json',
                   'training_output/dpp_parameters/training_cost.json']:
        if member in artifact.names:
            record = json.loads(artifact.read(member))
            if record.get('status', 'completed') != 'completed':
                raise ValueError('Incomplete fit: ' + str(artifact.path))
            epochs = record.get('epochs', record.get('epoch_wall_seconds_including_validation', []))
            if len(epochs) != 100:
                raise ValueError('Expected one complete 100-epoch fit: ' + str(artifact.path))
            return float(record['total_fit_seconds'])/3600, 'fit_callback', {'member':member, 'record':record}
    # Legacy P2/LIP did not record a fit timer. Use the main training logger's
    # file-creation event through its last event, explicitly marked as estimate.
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    prefix='training_output/lightning_logs/version_0/'
    members=[n for n in artifact.names if n.startswith(prefix)
             and Path(n).parent.as_posix()==prefix.rstrip('/') and 'events.out.tfevents' in n]
    if len(members)!=1:
        raise ValueError('No unique main training event file: ' + str(artifact.path))
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/Path(members[0]).name;path.write_bytes(artifact.read(members[0]))
        events=list(EventFileLoader(str(path)).Load())
    walls=[e.wall_time for e in events if e.wall_time>0]
    return (max(walls)-min(walls))/3600, 'TensorBoard_span_estimate', {
        'member':members[0], 'first_wall_time':min(walls), 'last_wall_time':max(walls),
        'scope':'main training logger creation to final recorded event; approximation to fit wall time'}


def export(workspace, output):
    output.mkdir(parents=True,exist_ok=True)
    baseline=workspace/'model/Loss_Models/pre-model/P2_Ex7_RepDWNet_WDice-Loss_artifacts.zip'
    entries=[('P2-WDice',baseline)]
    for family in ['DPP','AAA','LIP']:
        candidates=sorted((workspace/'model/Down_Models'/family).glob('*.zip'))
        if len(candidates)!=3:raise ValueError(f'Expected 3 {family} runs, got {candidates}')
        for path in candidates:
            placement=path.name.split('_')[3]
            entries.append((family+'-'+placement,path))
    rows=[]; provenance={'_profiling_protocol': {'torch':torch.__version__, 'device':'cpu', 'resolution':[768,768], 'batch_size':1, 'dtype':'float32', 'graph':'original_pre_deploy_eval', 'operators':'aten::conv2d', 'flops_per_mac':2}}
    for name,path in entries:
        print('[EXPORT]',name,flush=True)
        artifact=Artifact(path);config=artifact.config()
        member=artifact.checkpoint('best_mIoU')
        payload,state,fingerprint=checkpoint_payload(artifact,member)
        history=artifact.history().dropna(subset=['epoch'])
        selected=history[history.epoch==int(payload['epoch'])]
        if len(selected)!=1:raise ValueError('No unique selected history row')
        metric=selected.iloc[0]
        if abs(float(metric.mIoU)-float(history.mIoU.max()))>1e-8:
            raise ValueError('Selected checkpoint is not best recorded mIoU')
        if int(history.epoch.max())!=99 or config['train']['max_epoch']!=100:
            raise ValueError('Not a complete 100-epoch run')
        if config['backbone'].get('deploy',False):raise ValueError('Expected pre-deploy host')
        model=get_backbone(config).float().eval()
        model.load_state_dict(state,strict=True)
        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        with torch.inference_mode(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as prof:
            result=model(torch.zeros(1,3,768,768))
            if not torch.isfinite(result).all():raise ValueError('Non-finite checkpoint output')
        flops=sum(e.flops for e in prof.key_averages() if e.key=='aten::conv2d')
        if flops<=0:raise ValueError('Convolution profiler returned no FLOPs')
        hours,time_source,time_details=training_time(artifact)
        row={'Model':name,'Time-Training (h)':hours,'MACs (G)':flops/2/1e9,
             'FLOPs (G)':flops/1e9,'Params (M)':params/1e6,'Params (count)':params,
             **{column:float(metric[key])*100 for column,key in METRICS.items()},
             'Checkpoint epoch (0-based)':int(payload['epoch']),
             'Metric source':'stored_validation_best_mIoU','Time source':time_source}
        rows.append(row)
        provenance[name]={'artifact':str(path.relative_to(workspace)), 'checkpoint':member,
                          'checkpoint_fingerprint':fingerprint,'stages':config['backbone'].get('downsampling',{}),
                          'params_exact':params,'conv_macs_exact':flops//2,'conv_flops_exact':flops,
                          'metrics_unrounded':{k:row[k] for k in METRICS}, 'training_time':time_details}
        del model,payload,state,result,prof;gc.collect()
    # Table II entries belong to the same ablation. Table VI mean P/R/F1
    # must NOT be attached to the Table II theta=0.5 run.
    for theta,miou,soil,crop,weed in [(0,87.92,99.38,95.48,68.90),(.5,88.89,99.37,95.59,71.70)]:
        rows.append({'Model':f'RPD-Net (t={theta})','mIoU (%)':miou,'Soil IoU (%)':soil,
                     'Crop IoU (%)':crop,'Weed IoU (%)':weed,'Params (M)':.189,
                     'Metric source':'paper_Table_II','Time source':'not_reported'})
    target=output/'downsampling_model_comparison.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for row in rows:
            writer.writerow({key: ('N/A' if row.get(key) is None else
                f'{row[key]:.3f}' if isinstance(row[key],float) else row[key]) for key in FIELDS})
    (output/'downsampling_model_comparison_provenance.json').write_text(
        json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    (output/'downsampling_model_comparison_notes.md').write_text('''# Downsampling model comparison

- CSV: 12 models; numeric floats rounded to 3 decimals. Quality metrics are percentages, training time hours, Params millions, MACs/FLOPs billions (decimal G).
- All internal model metrics come from `training_output/val_history.csv` at the saved **best mIoU** checkpoint. These are stored training-validation results, not fresh inference from the unified evaluator. No metric is selected independently from a different epoch. Each run completed 100 epochs; epoch in CSV is zero-based.
- LIP Ex3 is **Middle [2,3]**, not All. DPP/AAA Ex3 are All [1,2,3,4].
- MACs/FLOPs were recomputed on the actual loaded checkpoint, original pre-deploy eval graph, batch 1, FP32, **768×768**, using executed `aten::conv2d` (includes functional RPD convolutions and fixed AAA blur convolutions). FLOPs = 2 × MACs. Scope excludes normalization, activations/gating, comparisons, pooling/weighted aggregation, elementwise arithmetic, padding and interpolation. This is convolution compute, not total operator FLOPs. Params count all trainable parameters on this same pre-deploy graph; fixed AAA filter buffers are excluded. The additional Params (count) column preserves exact internal-model parameter counts despite rounding Params (M) to 3 decimals. Paper exact counts remain N/A because only rounded millions are reported. See JSON for exact MAC/FLOP counts before display rounding.
- AAA/DPP time: observed `total_fit_seconds` callback, including validation, for the complete fit; excludes notebook setup, preflight, disposable benchmarks and ZIP export. P2-WDice/LIP have no fit timer: their values are **estimates** from main training TensorBoard file creation to final recorded event, including validation. These estimates may omit teardown and differ slightly from full fit time. `Time source` identifies the method. These times describe recorded runs, not a controlled hardware speed benchmark.
- RPD-Net t=0 and t=0.5: baseline paper **Table II**, following the earlier P2 comparison. They use a different training protocol; no corresponding local checkpoints are represented by these rows. Table II does not provide Weed Precision/Recall/F1 or training time; leave N/A. Params 0.189 M is the paper's rounded quantity.
- Do not mix paper Table II's t=0.5 metrics (mIoU 88.89, Weed IoU 71.70) with the mean results in Tables V/VI (mIoU 88.26, Weed IoU 70.03, Weed Precision 73.74, Recall 91.53, F1 81.68). The latter are a separate reported aggregate.
- Paper prose reports generic RPD-Net pre-deploy 13.45 G MACs at 768×768, but does not identify separate MAC measurements for both Table II ablations. Their MACs/FLOPs remain N/A rather than assigning that generic figure to both configurations.
- Source paper: `workspace/docs/main/references/Baseline_RPD_Learning_Efficient_Crops_and_Weeds_for_Field_Semantic_Segmentation_in_Drone_Images.pdf`. Artifact paths, checkpoint identities, exact counts and timer evidence are in the companion provenance JSON.
''',encoding='utf-8')
    print(target,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    export(args.workspace.resolve(),args.output or args.workspace/'docs/dlogs/P3_DefiRPD_2-DPD/Output')
