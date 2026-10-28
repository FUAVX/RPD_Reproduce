"""Training-interpreter bridge. The notebook kernel never imports Torch/NumPy."""
import argparse,importlib.metadata,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def doctor(allow_cpu=False):
    if sys.version_info[:2]!=(3,11):raise RuntimeError('Training requires isolated Python3.11')
    if sys.prefix==sys.base_prefix:raise RuntimeError('Training must use a virtual environment')
    from utils.tensorboard_compat import use_tensorboard_without_tensorflow
    use_tensorboard_without_tensorflow()
    import numpy as np
    import torch,torchvision
    import pytorch_lightning,torchmetrics
    import pandas,skimage,cv2,albumentations
    from utils.ndc_colab import configure_fp32
    configure_fp32()
    # Exercise compiled extensions and NumPy/Torch conversion, not metadata only.
    a=np.random.default_rng(0).normal(size=(2,3)).astype(np.float32)
    assert np.array_equal(torch.from_numpy(a).numpy(),a)
    torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]),torch.tensor([1.]),.5)
    required={'numpy':'1.26.4','torch':'2.5.1','torchvision':'0.20.1','pytorch-lightning':'1.9.5','torchmetrics':'0.11.4'}
    for name,version in required.items():
        actual=importlib.metadata.version(name).split('+')[0]
        if actual!=version:raise RuntimeError(f'{name} expected{version}, found{actual}')
    gpu=torch.cuda.is_available()
    if not allow_cpu and (not gpu or torch.cuda.device_count()!=1 or 'A100' not in torch.cuda.get_device_name(0) or torch.version.cuda!='12.4'):
        raise RuntimeError('Select one A100; require PyTorch2.5.1 CUDA12.4 wheels; no fallback')
    return {'status':'passed','python':sys.version,'interpreter':sys.executable,
            'isolated':sys.prefix!=sys.base_prefix,'ABI_probe':'passed','cuda_build':torch.version.cuda,
            'gpu_name':torch.cuda.get_device_name(0) if gpu else None,
            'gpu_vram_bytes':torch.cuda.get_device_properties(0).total_memory if gpu else None,
            'scope':'CPU_import_ABI_only' if allow_cpu else 'A100_import_ABI_and_device_check_not_training_memory',
            'versions':{name:importlib.metadata.version(name) for name in ['torch','torchvision','pytorch-lightning','torchmetrics','numpy','pandas','scikit-image','opencv-python-headless','albumentations','kagglehub'] if importlib.util.find_spec(name.replace('-','_')) is not None or name in ('pytorch-lightning','scikit-image','opencv-python-headless')}}

def dataset(handle):
    import kagglehub
    try:root=Path(kagglehub.dataset_download(handle))
    except Exception as error:
        if getattr(getattr(error,'response',None),'status_code',None) in (401,403):
            raise RuntimeError('Kaggle authentication required: add Colab Secret KAGGLE_API_TOKEN, grant access, then rerun. Never paste tokens into source.') from error
        raise
    def valid(p):return all((p/part).is_dir() for part in ['train/images','train/semantics','val/images','val/semantics'])
    candidates=[root,root/'PhenoBench',root/'PhenoBench-v110'/'PhenoBench']
    candidates.extend(p.parent for p in root.rglob('train') if p.is_dir())
    resolved=next((p for p in candidates if valid(p)),None)
    if resolved is None:raise FileNotFoundError('Downloaded dataset does not contain complete PhenoBench splits')
    return {'handle':handle,'dataset_dir':str(resolved)}

def checkpoint_info(checkpoint,config):
    import torch,yaml
    from utils.ndc_colab_io import PROTOCOL
    cfg=yaml.safe_load(Path(config).read_text());state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if state.get('execution_protocol')!=PROTOCOL or state.get('nested')!=cfg['backbone'].get('nested'):
        raise ValueError('Wrong model or execution protocol in checkpoint')
    return {'epoch':int(state['epoch']),'global_step':int(state['global_step']),
            'completed':state['epoch']==99 and state['global_step']==17600}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['doctor','dataset','checkpoint-info']);p.add_argument('--output',required=True)
    p.add_argument('--allow-cpu',action='store_true');p.add_argument('--handle');p.add_argument('--checkpoint');p.add_argument('--config');args=p.parse_args()
    if args.action=='doctor':result=doctor(args.allow_cpu)
    elif args.action=='dataset':result=dataset(args.handle)
    else:result=checkpoint_info(args.checkpoint,args.config)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
