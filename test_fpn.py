"""Fuse topology/recipe/gradient checks and actual trainer/CLI smoke entry points."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
import yaml
from models import get_backbone,get_criterion
from models.rpdnet.fuse_fpn import PyramidDecoder,OUTPUT_LEVELS,c2_xavier_fill
from models.rpdnet.RepDWNet import DoubleConv_down_repdw
from scripts.export_fpn import export,load_model
from scripts.fpn_smoke import module,trainer_smoke,pipeline_smoke
from utils.fpn_campaign import ROOT,SCENARIOS,config


class FPNTests(unittest.TestCase):
    def test_locked_recipe_and_common_encoder_initialization(self):
        baseline=yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        torch.manual_seed(baseline['seed']);control=get_backbone(config('control'))
        common={k:v for k,v in control.state_dict().items() if k.startswith(('in_conv.','down'))}
        for mode in SCENARIOS:
            cfg=config(mode);recipe=deepcopy(cfg);recipe['backbone'].pop('fusion',None);recipe['experiment']=baseline['experiment']
            self.assertEqual(recipe,baseline)
            torch.manual_seed(cfg['seed']);net=get_backbone(cfg)
            state=net.state_dict()
            for key,value in common.items():self.assertTrue(torch.equal(value,state[key]),(mode,key))
            self.assertEqual(sum(isinstance(m,nn.MaxPool2d) for m in net.modules()),4)
            self.assertFalse(any('aux_head' in k for k in state))
            for key in ['downsampling','deep_supervision']:
                invalid=deepcopy(cfg);invalid['backbone'][key]={}
                with self.assertRaises(ValueError):get_backbone(invalid)

    def test_control_exact_original_state_and_outputs(self):
        text=subprocess.check_output(['git','show','d77d518:models/rpdnet/RepDWNet.py'],cwd=ROOT,text=True)
        namespace={};exec(compile(text,'original_P2','exec'),namespace)
        torch.manual_seed(1);reference=namespace['RepDWNet'](3).eval()
        torch.manual_seed(1);actual=get_backbone(config('control')).eval()
        self.assertEqual(set(actual.state_dict()),set(reference.state_dict()))
        for k,v in actual.state_dict().items():self.assertTrue(torch.equal(v,reference.state_dict()[k]))
        for shape in [(32,32),(33,35)]:
            x=torch.randn(1,3,*shape)
            with torch.inference_mode():torch.testing.assert_close(actual(x),reference(x),rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in actual.parameters()),189827)

    def test_add_width_and_projection(self):
        for mode,projected in [('aex1',False),('aex2',True)]:
            net=get_backbone(config(mode)).eval()
            for stage,channels,out in [(1,128,64),(2,64,32),(3,32,16),(4,16,16)]:
                block=getattr(net,f'up{stage}')
                self.assertEqual((block.conv.in_channels,block.conv.mid_channels,block.conv.out_channels),(channels,channels,out))
                self.assertIsInstance(block.lateral,nn.Conv2d if projected else nn.Identity)
                x=torch.randn(1,channels,3,4);skip=torch.randn(1,channels,7,9)
                with torch.no_grad():
                    up=nn.functional.pad(block.up(x),[0,1,0,1])
                    expected=block.conv(up+block.lateral(skip))
                    torch.testing.assert_close(block(x,skip),expected,rtol=0,atol=0)

    def test_pyramid_exact_mean_unrefined_path_and_lazy_modules(self):
        features={i:torch.randn(1,c,s,s) for i,c,s in [(2,32,16),(3,64,8),(4,128,4),(5,128,2)]}
        for mode in [m for m in SCENARIOS if m.startswith('b')]:
            net=get_backbone(config(mode));decoder=net.fpn.eval();fusion=config(mode)['backbone']['fusion']
            levels=OUTPUT_LEVELS[fusion['pyramid']]
            self.assertEqual(set(decoder.laterals),{str(i) for i in range(min(levels),6)})
            self.assertEqual(set(decoder.refinements),{str(i) for i in levels if i!=5})
            self.assertFalse(any(hasattr(net,k) for k in ['up1','up2','up3','up4','out_conv']))
            self.assertFalse(any(isinstance(m,(nn.BatchNorm2d,nn.ReLU,nn.Dropout)) for m in decoder.modules()))
            with torch.inference_mode():
                merged=decoder.laterals['5'](features[5]);outputs=[]
                if 5 in levels:outputs.append(decoder.resize(merged,(16,16)))
                for i in range(4,min(levels)-1,-1):
                    merged=decoder.laterals[str(i)](features[i])+decoder.resize(merged,features[i].shape[-2:])
                    if i in levels:outputs.append(decoder.resize(decoder.refinements[str(i)](merged),(16,16)))
                expected=decoder.resize(decoder.head(torch.stack(outputs).mean(0)),(33,35))
                torch.testing.assert_close(decoder(features,(33,35)),expected,rtol=1e-5,atol=1e-6)
            for refinement in decoder.refinements.values():
                if fusion['refinement']=='dpw':
                    self.assertEqual(refinement[0].groups,32);self.assertIsNone(refinement[0].bias)
                    self.assertIsNotNone(refinement[1].bias)
                else:self.assertIsNotNone(refinement.bias)

    def test_new_conv_initialization_and_official_fixture(self):
        fixture=ROOT/'tests/reference/fpn/FPN.py'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),'1233464321577d37348566e672d4e590e3c185bddf6379a0c345cd416ab4fd65')
        self.assertIn('UpsampleNearest',fixture.read_text())
        conv=nn.Conv2d(8,16,3);torch.manual_seed(8);c2_xavier_fill(conv)
        expected=torch.empty_like(conv.weight);torch.manual_seed(8);nn.init.kaiming_uniform_(expected,a=1)
        self.assertTrue(torch.equal(conv.weight,expected));self.assertEqual(float(conv.bias.abs().sum()),0)
        for mode in SCENARIOS:
            net=get_backbone(config(mode))
            new=[net.up1.lateral,net.up2.lateral,net.up3.lateral,net.up4.lateral] if mode=='aex2' else list(net.fpn.modules()) if mode.startswith('b') else []
            for m in new:
                if isinstance(m,nn.Conv2d) and m.bias is not None:self.assertEqual(float(m.bias.abs().sum()),0)

    def test_all_variant_backward_and_checkpoint_identity(self):
        for mode in SCENARIOS:
            cfg=config(mode);net=get_backbone(cfg).train();x=torch.randn(2,3,33,35)
            y=torch.randint(0,3,(2,33,35));out=net(x)
            self.assertEqual(out.shape,(2,3,33,35))
            loss=get_criterion(cfg)(out,y,mode='train',mask_keep=y!=255);loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for name,p in net.named_parameters():
                self.assertIsNotNone(p.grad,(mode,name));self.assertTrue(torch.isfinite(p.grad).all(),(mode,name))
            with tempfile.TemporaryDirectory() as tmp:
                ckpt={'state_dict':{'network.'+k:v for k,v in net.state_dict().items()},'fusion':cfg['backbone'].get('fusion'),'val_history':[]}
                path=Path(tmp)/'weights.ckpt';torch.save(ckpt,path)
                report=export(cfg,path,Path(tmp)/'export',sizes=((32,32),(33,35)))
                self.assertEqual(report['fusion_preserved'],cfg['backbone'].get('fusion'))
                wrong=deepcopy(cfg);wrong['backbone']['fusion']={'variant':'add'} if mode!='aex1' else {'variant':'projected_add'}
                with self.assertRaises(ValueError):load_model(wrong,path)
                with self.assertRaises(ValueError):module(wrong).on_load_checkpoint(ckpt)
            for change in [{'channels':64},{'align_corners':False},{'normalization':'bn'}]:
                if mode.startswith('b'):
                    wrong=deepcopy(cfg);wrong['backbone']['fusion'].update(change)
                    with self.assertRaises(ValueError):get_backbone(wrong)

    def test_original_training_objective_and_optimizer(self):
        original=subprocess.check_output(['git','show','d77d518:models/model_multimetrics.py'],cwd=ROOT,text=True)
        actual=(ROOT/'models/model_multimetrics.py').read_text()
        def body(source,name):
            tree=ast.parse(source);return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ['training_step','on_train_epoch_end','compute_loss','configure_optimizers','lr_scaling']:
            self.assertEqual(body(actual,name),body(original,name))

    def test_dataset_guard_blocks_unknown_and_ignore255(self):
        from scripts.check_fpn_data import audit_dataset
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            for split in ['train','val']:
                for folder in ['images','semantics']:(Path(tmp)/split/folder).mkdir(parents=True)
                Image.fromarray(np.zeros((2,2,3),dtype=np.uint8)).save(Path(tmp)/split/'images/a.png')
                Image.fromarray(np.array([[0,1],[2,4]],dtype=np.uint8)).save(Path(tmp)/split/'semantics/a.png')
            self.assertEqual(audit_dataset(tmp)['status'],'passed')
            for label,status in [(255,'blocked_pending_ignore255_decision'),(9,'blocked_unknown_labels')]:
                Image.fromarray(np.array([[0,1],[2,label]],dtype=np.uint8)).save(Path(tmp)/'val/semantics/a.png')
                self.assertEqual(audit_dataset(tmp)['status'],status)

    def test_notebook_modes_and_syntax(self):
        nb=json.loads((ROOT/'notebook/kaggle_train_repdwnet_fpn.ipynb').read_text());namespace={}
        exec(''.join(nb['cells'][1]['source']),namespace)
        self.assertEqual(namespace['EXPERIMENTS'],SCENARIOS);self.assertEqual(namespace['GIT_REF'],'f/fuse-fpn')
        self.assertEqual(namespace['DEVICES'],2)
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(line for line in c['source'] if not line.lstrip().startswith('!')))
                self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])


if __name__=='__main__':
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--pipeline-smoke',action='store_true')
    parser.add_argument('--config');parser.add_argument('--output-dir');parser.add_argument('--accelerator',default='cpu')
    parser.add_argument('--devices',type=int,default=1);parser.add_argument('--resume',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.smoke or args.pipeline_smoke:
        if not args.config or not args.output_dir:parser.error('config/output-dir required')
        cfg=yaml.safe_load(Path(args.config).read_text())
        if args.smoke:trainer_smoke(cfg,args.output_dir,args.accelerator,args.devices,args.resume)
        else:pipeline_smoke(cfg,args.output_dir,args.accelerator,args.devices)
    else:unittest.main(argv=[__file__,*remaining])
