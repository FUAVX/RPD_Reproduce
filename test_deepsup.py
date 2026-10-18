"""DeepSup correctness + actual Lightning train/checkpoint/resume smoke.

python test_deepsup.py
python test_deepsup.py --smoke --config configs/p3_deepsup/P3_DeepSup_Ex2_x4.yaml --output-dir RUN --accelerator gpu --devices 2
"""
import argparse
import ast
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import yaml
from models import get_backbone, get_criterion
from models.model_multimetrics import SegmentationNetwork
from models.rpdnet.deep_supervision import AuxiliaryHead, TAP_CHANNELS
from callbacks.config_callback import ConfigCallback
from callbacks.deepsup_cost import DeepSupCostCallback
from scripts.export_deepsup import export, load_training_model

ROOT=Path(__file__).resolve().parent
SCENARIOS={'control':('P3_DeepSup_Control',None,0),
           'ex1':('P3_DeepSup_Ex1_x3','x3',18595),'ex2':('P3_DeepSup_Ex2_x4','x4',74051),
           'ex3':('P3_DeepSup_Ex3_x5','x5',74051),'ex4':('P3_DeepSup_Ex4_x6','x6',18595),
           'ex5':('P3_DeepSup_Ex5_x7','x7',4691)}
def config(key):return yaml.safe_load((ROOT/'configs/p3_deepsup'/f'{SCENARIOS[key][0]}.yaml').read_text())
def module(cfg):
    return SegmentationNetwork(get_backbone(cfg),get_criterion(cfg),cfg['train']['learning_rate'],cfg['train']['weight_decay'],
        train_step_settings=['regular'],val_step_settings=['regular'],scheduler_type=cfg['train']['scheduler'],warmup_epochs=cfg['train']['warmup_epochs'])

class Synthetic(Dataset):
    def __len__(self):return 8
    def __getitem__(self,i):
        g=torch.Generator().manual_seed(i)
        return {'input_image':torch.randn(3,32,32,generator=g),'anno':torch.randint(0,3,(32,32),generator=g)}

def trainer_smoke(cfg,output,accelerator='cpu',devices=1,resume=False):
    pl.seed_everything(cfg['seed'])
    net=module(cfg);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    logger=TensorBoardLogger(str(output),name='lightning_logs',version=0,default_hp_metric=False)
    ckpt=ModelCheckpoint(dirpath=str(output/'checkpoints'),save_last=True,monitor='val_mIoU',mode='max')
    trainer=pl.Trainer(accelerator=accelerator,devices=devices,
        strategy=DDPStrategy(find_unused_parameters=False,timeout=timedelta(seconds=90)) if devices>1 else 'auto',
        sync_batchnorm=False,precision=32,max_epochs=2 if resume else 1,
        limit_train_batches=2,limit_val_batches=1,num_sanity_val_steps=2,enable_progress_bar=False,
        enable_model_summary=False,log_every_n_steps=1,default_root_dir=str(output),logger=logger,
        callbacks=[ConfigCallback(cfg),ckpt,DeepSupCostCallback(output)])
    loader=DataLoader(Synthetic(),batch_size=2)
    trainer.fit(net,loader,loader,ckpt_path=str(output/'checkpoints/last.ckpt') if resume else None)
    if trainer.is_global_zero:
        state=torch.load(output/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert state['global_step']==(4 if resume else 2)
        import csv
        expected_epochs = [0, 1] if resume else [0]
        with (output/'val_history.csv').open() as stream: validation_rows=list(csv.DictReader(stream))
        assert [int(row['epoch']) for row in validation_rows] == expected_epochs
        assert [row['epoch'] for row in state['val_history']] == expected_epochs
        assert net.val_history == state['val_history']
        assert state['optimizer_states'] and all(torch.isfinite(v).all() for v in state['state_dict'].values())
        if cfg['backbone'].get('deep_supervision'):
            assert any(k.startswith('network.aux_head.') for k in state['state_dict'])
            import csv
            with (output/'deep_supervision_history.csv').open() as stream:history=list(csv.DictReader(stream))
            assert len(history)==(2 if resume else 1)
            for row in history:
                assert abs(float(row['train_total_loss'])-(float(row['train_main_loss'])+.4*float(row['train_aux_loss'])))<1e-3
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events=EventAccumulator(logger.log_dir).Reload()
            for name in ['train_main_loss','train_aux_loss','train_total_loss','val_mIoU']:assert events.Scalars(name)
        load_training_model(cfg,output/'checkpoints/last.ckpt')
        cost=json.loads((output/'training_cost.json').read_text());assert cost['world_size']==devices
        assert cost['main_trainable_params']==189827
        print(f'PASS trainer {cfg["experiment"]["id"]}: devices={devices}, resume={resume}',flush=True)

def pipeline_smoke(cfg, output, accelerator='cpu', devices=1):
    """Actual trainer CLI + PDC loader on explicitly synthetic32 crops/1 epoch."""
    import subprocess, sys
    from PIL import Image
    output=Path(output);dataset=output/'synthetic_data'
    rng=np.random.default_rng(cfg['seed'])
    for split,count in [('train',8),('val',4),('test',0)]:
        for folder in ['images','semantics']:(dataset/split/folder).mkdir(parents=True,exist_ok=True)
        for i in range(count):
            Image.fromarray(rng.integers(0,256,(32,32,3),dtype=np.uint8)).save(dataset/split/'images'/f'synthetic_{i}.png')
            Image.fromarray(rng.integers(0,5,(32,32),dtype=np.uint8)).save(dataset/split/'semantics'/f'synthetic_{i}.png')
    smoke=deepcopy(cfg);smoke['data']['path_to_dataset']=str(dataset.resolve());smoke['data']['num_workers']=0
    smoke['train'].update(max_epoch=1,batch_size=2,accelerator=accelerator,devices=devices)
    if accelerator=='cpu' and devices>1:smoke['train']['strategy']='ddp'
    smoke['train']['geometric_data_augmentations']['random_crop']={'height':32,'width':32}
    smoke['val'].update(batch_size=2,check_val_every_n_epoch=1)
    smoke['val']['geometric_data_augmentations']['center_crop']={'height':32,'width':32}
    config_path=output/'synthetic_config.yaml';config_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
    subprocess.run([sys.executable,str(ROOT/'multi_metric_train.py'),'--config',str(config_path),
        '--export_dir',str(output/'training_output'),'--devices',str(devices)],cwd=ROOT,check=True)
    assert (output/'training_output/checkpoints/last.ckpt').is_file()
    assert (output/'training_output/val_history.csv').is_file()
    if smoke['backbone'].get('deep_supervision'):assert (output/'training_output/deep_supervision_history.csv').is_file()
    assert json.loads((output/'training_output/training_cost.json').read_text())['world_size']==devices
    from scripts.check_deepsup_run import audit_run
    audit=audit_run(output/'training_output',expected_epochs=1,val_interval=1)
    (output/'campaign_audit.json').write_text(json.dumps(audit,indent=2))
    print('PASS actual trainer CLI + PDC loader, synthetic smoke only',flush=True)

class DeepSupTests(unittest.TestCase):
    def test_official_topology_and_fixed_settings(self):
        source=ast.parse((ROOT/'tests/reference/deepsup/pspnet.py').read_text())
        assignment=next(n for n in ast.walk(source) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='aux' for t in n.targets))
        reference=eval(compile(ast.Expression(assignment.value),'official_aux','eval'),{'nn':nn,'dropout':.1,'classes':3})
        for channels in [32,64,128]:
            head=AuxiliaryHead(channels)
            self.assertEqual([type(m) for m in head],[type(m) for m in reference])
            self.assertEqual(head[0].kernel_size,reference[0].kernel_size);self.assertEqual(head[0].padding,reference[0].padding)
            self.assertIsNone(head[0].bias);self.assertIsNotNone(head[-1].bias)
            self.assertEqual(head[0].out_channels,channels//2);self.assertEqual(head[1].eps,reference[1].eps)
            self.assertEqual(head[3].p,reference[3].p)
        for setting,value in [('weight',.5),('dropout',.2),('hidden_ratio',.25),('align_corners',False),('tap','x8'),('temperature',1)]:
            cfg=config('ex1');cfg['backbone']['deep_supervision'][setting]=value
            with self.assertRaises(ValueError):get_backbone(cfg)

    def test_configs_host_initialization_rng_and_main_identity(self):
        base=yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        for key,(_,tap,added) in SCENARIOS.items():
            cfg=config(key);inherited=deepcopy(cfg);inherited['backbone'].pop('deep_supervision',None);inherited['experiment']=base['experiment']
            self.assertEqual(inherited,base)
            torch.manual_seed(cfg['seed']);control=get_backbone(base).train();expected={k:v.clone() for k,v in control.state_dict().items()};rng=torch.get_rng_state().clone()
            torch.manual_seed(cfg['seed']);model=get_backbone(cfg).train();self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for name,value in expected.items():self.assertTrue(torch.equal(value,model.state_dict()[name]),(key,name))
            self.assertEqual(sum(p.numel() for p in model.parameters()),189827+added)
            self.assertEqual(sum(isinstance(m,AuxiliaryHead) for m in model.modules()),0 if tap is None else 1)
            x=torch.randn(2,3,33,35)
            with torch.no_grad():
                expected_output=control(x)
                actual=model(x,return_aux=True)['main'] if tap else model(x)
            torch.testing.assert_close(expected_output,actual,rtol=0,atol=0)
            self.assertEqual(sum(isinstance(getattr(model,f'down{i}')[0],nn.MaxPool2d) for i in range(1,5)),4)

    def test_exact_taps_raw_logits_and_dynamic_resize(self):
        modules={'x3':'down2','x4':'down3','x5':'down4','x6':'up1','x7':'up2'}
        for key,(_,tap,_) in list(SCENARIOS.items())[1:]:
            model=get_backbone(config(key)).train();seen={}
            handles=[getattr(model,modules[tap]).register_forward_hook(lambda m,i,o:seen.update(tap=o)),
                     model.aux_head.register_forward_pre_hook(lambda m,i:seen.update(input=i[0])),
                     model.aux_head.register_forward_hook(lambda m,i,o:seen.update(raw=o))]
            try:
                out=model(torch.randn(2,3,65,67),return_aux=True,target_size=(63,69))
                self.assertIs(seen['tap'],seen['input']);self.assertEqual(seen['input'].shape[1],TAP_CHANNELS[tap])
                downscale={'x3':4,'x4':8,'x5':16,'x6':8,'x7':4}[tap]
                self.assertEqual(tuple(seen['input'].shape[-2:]),(65//downscale,67//downscale))
                self.assertEqual(tuple(out['main'].shape),(2,3,65,67));self.assertEqual(tuple(out['aux'].shape),(2,3,63,69))
                torch.testing.assert_close(out['aux'],F.interpolate(seen['raw'],size=(63,69),mode='bilinear',align_corners=True),rtol=0,atol=0)
                model.eval()
                def forbidden(*args):raise AssertionError('Aux head executed during inference')
                h=model.aux_head.register_forward_pre_hook(forbidden)
                with torch.inference_mode():self.assertIsInstance(model(torch.randn(1,3,32,32)),torch.Tensor)
                h.remove()
            finally:
                for handle in handles:handle.remove()

    def test_same_criterion_total_once_and_aux_gradient_reach(self):
        downstream={'x3':['down3','down4','up1','up2','up3','up4','out_conv'],
                    'x4':['down4','up1','up2','up3','up4','out_conv'],
                    'x5':['up1','up2','up3','up4','out_conv'],
                    'x6':['up2','up3','up4','out_conv'],'x7':['up3','up4','out_conv']}
        for key,(_,tap,_) in list(SCENARIOS.items())[1:]:
            cfg=config(key);net=module(cfg).train();net._trainer=SimpleNamespace(current_epoch=0)
            captured={};h=net.network.register_forward_hook(lambda m,i,o:captured.update(o) if isinstance(o,dict) else None)
            batch=next(iter(DataLoader(Synthetic(),batch_size=2)))
            result=net.training_step(batch,0);h.remove()
            main=net.criterion(captured['main'],batch['anno'],mode='train',mask_keep=batch['anno']!=255)
            aux=net.criterion(captured['aux'],batch['anno'],mode='train',mask_keep=batch['anno']!=255)
            torch.testing.assert_close(result['loss'],main+.4*aux,rtol=0,atol=0)
            net.zero_grad(set_to_none=True);aux.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in net.network.in_conv.parameters()),tap)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.network.aux_head.parameters()))
            for name in downstream[tap]:self.assertTrue(all(p.grad is None for p in getattr(net.network,name).parameters()),(tap,name))
            optimizers,_=net.configure_optimizers();optimizer=optimizers[0]
            ids={id(p) for group in optimizer.param_groups for p in group['params']}
            self.assertTrue(all(id(p) in ids for p in net.network.aux_head.parameters()))
            self.assertEqual(optimizer.defaults['lr'],cfg['train']['learning_rate'])
            self.assertEqual(optimizer.defaults['weight_decay'],cfg['train']['weight_decay'])

    def test_ignore_mask_criterion_and_dataset_pending_guard(self):
        from scripts.check_deepsup_data import audit_dataset
        from PIL import Image
        criterion=get_criterion(config('ex1'));y=torch.tensor([[[255,0],[1,2]]])
        x=torch.randn(1,3,2,2,requires_grad=True);loss=criterion(x,y,mode='train',mask_keep=y!=255);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(x.grad[0,:,0,0].abs().sum(),0)
        with tempfile.TemporaryDirectory() as tmp:
            for split in ['train','val']:
                for folder in ['images','semantics']:(Path(tmp)/split/folder).mkdir(parents=True)
                Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(Path(tmp)/split/'images/a.png')
                Image.fromarray(np.array([[0,1],[2,255]],dtype=np.uint8)).save(Path(tmp)/split/'semantics/a.png')
            self.assertEqual(audit_dataset(tmp)['status'],'blocked_pending_ignore255_decision')

    def test_training_checkpoint_resume_and_stripped_export(self):
        for key in SCENARIOS:
            with self.subTest(mode=key),tempfile.TemporaryDirectory() as tmp:
                cfg=config(key);trainer_smoke(cfg,tmp);trainer_smoke(cfg,tmp,resume=True)
                report=export(cfg,Path(tmp)/'checkpoints/last.ckpt',Path(tmp)/'main_export',sizes=((32,32),(33,35)))
                self.assertTrue(report['auxiliary_removed']);self.assertEqual(report['main_trainable_params'],189827)
                bare_cfg=yaml.safe_load((Path(tmp)/'main_export/main_config.yaml').read_text())
                self.assertNotIn('deep_supervision',bare_cfg['backbone'])
                bare=get_backbone(bare_cfg).eval();bare.load_state_dict(torch.load(Path(tmp)/'main_export/main_state_dict.pt',weights_only=True),strict=True)
                trained=load_training_model(cfg,Path(tmp)/'checkpoints/last.ckpt')
                x=torch.randn(1,3,33,35)
                with torch.inference_mode():torch.testing.assert_close(trained(x),bare(x),rtol=0,atol=0)

    def test_checkpoint_tap_metadata_rejects_same_width_wrong_placement(self):
        cfg=config('ex1');net=get_backbone(cfg)
        payload={'state_dict':{'network.'+k:v for k,v in net.state_dict().items()},
                 'deep_supervision':cfg['backbone']['deep_supervision']}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.ckpt';torch.save(payload,path)
            load_training_model(cfg,path)
            with self.assertRaisesRegex(ValueError,'configuration'):
                load_training_model(config('ex4'),path)  # x3/x6 share C64 shapes
            with self.assertRaisesRegex(ValueError,'configuration'):
                module(config('ex4')).on_load_checkpoint(payload)

    def test_failed_fusion_is_not_exported_or_silently_tolerated(self):
        from unittest.mock import patch
        class ChangedGraph(nn.Module):
            def __init__(self, model):
                super().__init__();self.model=model
            def forward(self, x):return self.model(x)+1.0
        cfg=config('ex2')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'training.pt';torch.save(get_backbone(cfg).state_dict(),path)
            out=Path(tmp)/'export'
            with patch('scripts.export_deepsup.RPD_model_deploy',side_effect=lambda m,do_copy:ChangedGraph(m)):
                with self.assertRaises(RuntimeError):export(cfg,path,out,fuse=True,sizes=((32,32),))
            report=json.loads((out/'export_report.json').read_text())
            self.assertEqual(report['fusion']['status'],'failed')
            self.assertEqual(report['fusion']['absolute_logit_tolerance'],1e-4)
            self.assertTrue((out/'main_state_dict.pt').exists());self.assertFalse((out/'fused_state_dict.pt').exists())
            with self.assertRaises(FileExistsError):export(cfg,path,out,sizes=((32,32),))

    def test_rng_isolation_benchmark_exception(self):
        from scripts.benchmark_deepsup import isolated_rng
        import random
        state=torch.get_rng_state().clone();python_state=random.getstate();np_state=np.random.get_state();backend=torch.backends.cudnn.benchmark
        with self.assertRaises(RuntimeError):
            with isolated_rng(torch.device('cpu')):
                torch.rand(3);np.random.rand();random.random();torch.backends.cudnn.benchmark=not backend;raise RuntimeError('probe')
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertEqual(python_state,random.getstate());np.testing.assert_array_equal(np_state[1],np.random.get_state()[1]);self.assertEqual(torch.backends.cudnn.benchmark,backend)

    def test_notebook_modes_and_launch(self):
        nb=json.loads((ROOT/'notebook/kaggle_train_repdwnet_deepsup.ipynb').read_text());namespace={}
        exec(''.join(nb['cells'][1]['source']),namespace)
        self.assertEqual(namespace['GIT_REF'],'f/DeepSup');self.assertEqual(namespace['DEVICES'],2)
        for key,(name,tap,_) in SCENARIOS.items():self.assertEqual(namespace['EXPERIMENTS'][key],(name,tap))
        source='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
        for token in ['check_deepsup_data.py',"'--devices', str(DEVICES)",'benchmark_deepsup.py','export_deepsup.py']:
            self.assertIn(token,source)
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(line for line in c['source'] if not line.lstrip().startswith('!')))
                self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])

if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--smoke',action='store_true');parser.add_argument('--native-smoke',action='store_true');parser.add_argument('--pipeline-smoke',action='store_true')
    parser.add_argument('--config');parser.add_argument('--output-dir');parser.add_argument('--accelerator',default='cpu');parser.add_argument('--devices',type=int,default=1);parser.add_argument('--resume',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.pipeline_smoke:
        if not args.config or not args.output_dir:parser.error('pipeline smoke requires config/output-dir')
        pipeline_smoke(yaml.safe_load(Path(args.config).read_text()),args.output_dir,args.accelerator,args.devices)
    elif args.smoke:
        if not args.config or not args.output_dir:parser.error('smoke needs config and output-dir')
        trainer_smoke(yaml.safe_load(Path(args.config).read_text()),args.output_dir,args.accelerator,args.devices,args.resume)
    elif args.native_smoke:
        if not torch.cuda.is_available():raise RuntimeError('Native GPU smoke requires CUDA')
        for key in SCENARIOS:
            model=get_backbone(config(key)).cuda().train()
            with torch.no_grad():
                for size in [768,1024]:
                    out=model(torch.randn(1,3,size,size,device='cuda'),return_aux=True) if model.aux_head is not None else model(torch.randn(1,3,size,size,device='cuda'))
                    for value in out.values() if isinstance(out,dict) else [out]:assert torch.isfinite(value).all() and value.shape==(1,3,size,size)
            del model;torch.cuda.empty_cache()
        print('PASS native GPU shapes all six modes')
    else:unittest.main(argv=[__file__,*remaining])
