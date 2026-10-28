"""Notebook-side config/backup operations: no NumPy, Torch or Lightning imports."""
from copy import deepcopy
import hashlib,json,os,shutil,tempfile,zipfile
from pathlib import Path

PROTOCOL='ndc_colab_paired_batch4_v1'

def validate_recipe(cfg):
    from utils.ndc_campaign import config
    mode='control' if cfg['backbone'].get('nested') is None else 'ex1'
    expected=config(mode);actual=deepcopy(cfg);execution=actual.pop('execution')
    if execution['protocol']!=PROTOCOL:raise ValueError('Unknown Colab execution protocol')
    for key in ['path_to_dataset','num_workers']:
        expected['data'][key]=actual['data'][key]
    expected['train']['devices']=1;expected['train']['accumulate_grad_batches']=2
    expected['train']['accelerator']=actual['train'].get('accelerator','gpu')
    if execution.get('smoke_evidence_only'):
        if expected['train']['accelerator']!='cpu':raise ValueError('Smoke overrides are CPU-only')
        for key in ['max_epoch']:expected['train'][key]=actual['train'][key]
        expected['train']['geometric_data_augmentations']['random_crop']=actual['train']['geometric_data_augmentations']['random_crop']
        expected['val']['check_val_every_n_epoch']=actual['val']['check_val_every_n_epoch']
    if actual!=expected:raise ValueError('Colab recipe drift outside approved device/paired execution/data-path overrides')
    if cfg['train']['devices']!=1 or cfg['train']['accumulate_grad_batches']!=2:raise ValueError('Single GPU +2 paired microbatches required')
    return mode

def publish_backup(source,backup_dir):
    """Two rotating ZIPs; publish pointer only after a complete checksum-verified copy."""
    source=Path(source);backup=Path(backup_dir);backup.mkdir(parents=True,exist_ok=True)
    source_commit=(source/'source_commit.txt').read_text().strip()
    if len(source_commit)!=40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Backup must record a valid exact source commit SHA')
    pointer=backup/'latest.json'
    previous=json.loads(pointer.read_text()) if pointer.exists() else {}
    slot='B' if previous.get('slot')=='A' else 'A'
    with tempfile.TemporaryDirectory() as tmp:
        archive=Path(shutil.make_archive(str(Path(tmp)/'snapshot'),'zip',root_dir=source))
        digest=hashlib.sha256(archive.read_bytes()).hexdigest();target=backup/f'snapshot_{slot}.zip'
        shutil.copyfile(archive,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise IOError('Drive backup checksum mismatch')
    meta={'slot':slot,'archive':target.name,'sha256':digest,'protocol':PROTOCOL,
          'source_commit':source_commit}
    temporary=backup/'latest.json.tmp';temporary.write_text(json.dumps(meta,indent=2)+'\n');os.replace(temporary,pointer)
    return meta

def restore_backup(backup_dir,destination):
    backup=Path(backup_dir);meta=json.loads((backup/'latest.json').read_text())
    if meta['protocol']!=PROTOCOL:raise ValueError('Backup protocol mismatch')
    archive=backup/meta['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['sha256']:raise IOError('Backup checksum mismatch')
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Restore destination must be empty')
    with zipfile.ZipFile(archive) as stream:
        for name in stream.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Invalid archive path')
        stream.extractall(destination)
    return meta
