"""AAA profile for the unified evaluator; trained artifacts only, no random fallback.

Shares the DPP evaluator's actual global/lighting/GT-instance metrics and native
inference runner. AAA adds fixed-filter audits, native1024 timing and offline WSC.
"""
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
import torch

import dpp_evaluation as common


def discover_artifacts(workspace):
    root = Path(workspace)
    control = root/'model/Loss_Models/pre-model/P2_Ex7_RepDWNet_WDice-Loss_artifacts.zip'
    if not control.exists():control = control.with_suffix('')
    if not control.exists():raise FileNotFoundError(control)
    entries = [{'name': 'RepDWNet+WDice', 'artifact': str(control), 'stages': [], 'family': 'AAA'}]
    for ex, label, stages in [(1,'Early',[1,2]),(2,'Late',[3,4]),(3,'All',[1,2,3,4])]:
        matches = list((root/'model/Down_Models/AAA').glob(f'P3_AAA_Ex{ex}_{label}_*_artifacts.zip'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one completed AAA {label} ZIP in model/Down_Models/AAA; found {matches}. Train/export first, or resolve duplicate runs explicitly.')
        entries.append({'name': 'AAA-'+label, 'artifact': str(matches[0]), 'stages': stages, 'family': 'AAA'})
    return entries


def prepare_entries(workspace, selections=('best_mIoU','best_weedIoU')):
    entries = []
    for entry in discover_artifacts(workspace):
        artifact = common.Artifact(entry['artifact']);cfg = artifact.config()
        actual = cfg['backbone'].get('downsampling', {})
        if actual.get('stages', []) != entry['stages']:
            raise ValueError('AAA checkpoint placement/config mismatch.')
        if entry['stages'] and actual != {'variant':'aaa_tri3','stages':entry['stages'],
                'filt_size':3,'pad_type':'reflect','stride':2,'pad_off':0}:
            raise ValueError('Unexpected AAA-Now operator settings.')
        for selection in selections:
            member = artifact.checkpoint(selection)
            payload, _, fingerprint = common.checkpoint_payload(artifact, member)
            entries.append({**entry,'selection':selection,'member':member,'epoch':int(payload['epoch']),
                            'fingerprint':fingerprint,'config':cfg})
    common.recipe_audit(entries)
    return entries


def filter_analysis(workspace, entries, output):
    common.load_source(workspace)
    from models.rpdnet.aaa import tri3_filter
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in entries:
        artifact = common.Artifact(entry['artifact'])
        _, state, _ = common.checkpoint_payload(artifact, entry['member'])
        expected_keys = {f'down{s}.0.filt' for s in entry['stages']}
        found = {k for k in state if k.endswith('.filt')}
        if found != expected_keys:raise ValueError('AAA filter stage keys differ from declared topology.')
        for stage in entry['stages']:
            value = state[f'down{stage}.0.filt'];channels = 16*2**(stage-1)
            if not torch.equal(value, tri3_filter(channels, value.dtype, value.device)):
                raise ValueError('Checkpoint changed the fixed Tri-3 filter.')
            rows.append({'model':entry['name'],'selection':entry['selection'],'stage':stage,
                         'channels':channels,'filt_size':3,'buffer_elements':value.numel(),
                         'buffer_bytes':value.numel()*value.element_size(),'aaa_trainable_params':0,
                         'fixed_filter_matches':True})
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'aaa_filter_audit.csv', index=False)
    return frame


def run_wsc(workspace, entries, quality_metadata, dataset_root, output, device, mode):
    common.load_source(workspace)
    from scripts.analyze_aaa import run_analysis
    output = Path(output);rows = [];summaries = {}
    for entry in entries:
        key = entry['fingerprint']
        if key not in summaries:
            quality = quality_metadata['results'][key]
            cache = output/'wsc'/mode/key;cache.mkdir(parents=True, exist_ok=True)
            summary_path = cache/'wsc_summary.json'
            signature = {'quality_signature':quality['signature'], 'graph':'unfused_eval',
                'analysis_source_sha256':hashlib.sha256((Path(workspace)/'RPD_P2/scripts/analyze_aaa.py').read_bytes()).hexdigest()}
            old = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if old.get('evaluation_signature') == signature and old.get('status') == 'completed':
                summary = old
                print('[WSC CACHE]',entry['name'],flush=True)
            else:
                artifact = common.Artifact(entry['artifact'])
                names = None
                if mode == 'calibration':
                    names = pd.read_csv(Path(quality['cache_dir'])/'per_image.csv').image.tolist()
                with tempfile.TemporaryDirectory(prefix='aaa_wsc_') as temp:
                    checkpoint = Path(temp)/'selected.ckpt';checkpoint.write_bytes(artifact.read(entry['member']))
                    summary = run_analysis(entry['config'], checkpoint, dataset_root, cache,
                                           device=str(device), filenames=names)
                summary['checkpoint'] = entry['artifact']+'::'+entry['member']
                summary['checkpoint_fingerprint'] = key
                summary['evaluation_signature'] = signature
                common._json_write(summary_path, summary)
            summaries[key] = summary
        summary = summaries[key]
        rows.append({'model':entry['name'],'selection':entry['selection'],'scope':mode,
                     **{k:summary[k] for k in ['mean_wsc','std_wsc_across_images','images_seen','evaluated_images',
                                               'skipped_images','total_image_offset_pairs','skipped_image_offset_pairs','skipped_pairs_percent']},
                     **{name+'_wsc':v['mean_wsc'] for name,v in summary['per_direction'].items()}})
    frame = pd.DataFrame(rows)
    for selection in frame.selection.unique():
        mask = frame.selection == selection
        base = frame[mask & (frame.model == 'RepDWNet+WDice')].iloc[0].mean_wsc
        frame.loc[mask,'mean_wsc_delta_vs_WDice'] = frame.loc[mask,'mean_wsc']-base if pd.notna(base) else None
    frame.to_csv(output/f'{mode}_wsc_comparison.csv', index=False)
    return frame, summaries


def run_master_unified_evaluation(dataset_root, model_entries, output_dir, workspace=None,
                                 device=None, mode='full', run_wsc_analysis=True):
    workspace = Path(workspace or common.workspace_root());output = Path(output_dir)
    device = torch.device(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    frame, metadata = common.run_master_unified_evaluation(dataset_root,model_entries,output,workspace,device,mode)
    rows = []
    for entry in model_entries:
        result = metadata['results'][entry['fingerprint']]
        timing = result.get('latency_native1024', {})
        rows.append({'model':entry['name'],'selection':entry['selection'],
                     'aaa_buffer_elements':result['profile'].get('aaa_buffer_elements',0),
                     'aaa_learnable_params':result['profile'].get('aaa_learnable_params',0),
                     'native1024_forward_ms':timing.get('mean_latency_ms'),
                     'native1024_forward_p50_ms':timing.get('p50_latency_ms'),
                     'native1024_forward_p95_ms':timing.get('p95_latency_ms'),
                     'native1024_forward_fps':timing.get('throughput_fps')})
    pd.DataFrame(rows).to_csv(output/f'{mode}_aaa_efficiency.csv',index=False)
    metadata['family'] = 'AAA';metadata['wsc_requested'] = run_wsc_analysis
    if run_wsc_analysis:
        _, summaries = run_wsc(workspace,model_entries,metadata,dataset_root,output,device,mode)
        metadata['wsc_results'] = summaries
        metadata['wsc_full_validation_complete'] = mode == 'full' and all(s['full_validation_complete'] for s in summaries.values())
    common._json_write(output/f'{mode}_evaluation_manifest.json',metadata)
    return frame,metadata
