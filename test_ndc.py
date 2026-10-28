"""NDC-Full topology, Keras-like H semantics, P2 compatibility and pipeline checks."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
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
from models.rpdnet.RepDWNet import Up_repdw
from models.rpdnet.RPD_Module import RPD_model_deploy
from models.rpdnet.fuse_ndc import HUnit,upsample_like,KERAS_TRUNCATED_NORMAL_STD,SETTINGS
from utils.ndc_campaign import ROOT,SCENARIOS,config
from scripts.ndc_smoke import module,trainer_smoke,pipeline_smoke
from scripts.export_ndc import export,load_model


class NDCTests(unittest.TestCase):
    def test_control_exact_original_and_common_initialization_rng(self):
        text=subprocess.check_output(['git','show','d77d518:models/rpdnet/RepDWNet.py'],cwd=ROOT,text=True)
        namespace={};exec(compile(text,'P2_original','exec'),namespace)
        torch.manual_seed(1682409321);original=namespace['RepDWNet'](3).eval();rng=torch.get_rng_state().clone()
        for mode in SCENARIOS:
            torch.manual_seed(1682409321);net=get_backbone(config(mode)).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for k,v in original.state_dict().items():
                if mode=='control' or k.startswith(('in_conv.','down','up1.','out_conv.')):
                    self.assertTrue(torch.equal(v,net.state_dict()[k]),k)
            if mode=='control':
                self.assertEqual(set(original.state_dict()),set(net.state_dict()))
                for h,w in [(32,32),(33,35)]:
                    x=torch.randn(1,3,h,w)
                    with torch.inference_mode():torch.testing.assert_close(original(x),net(x),rtol=0,atol=0)
        # Old Up accepts both the original tensor and an equivalent one-item list.
        up=Up_repdw(128,32).eval();x=torch.randn(2,64,3,4);skip=torch.randn(2,64,7,9)
        with torch.no_grad():torch.testing.assert_close(up(x,skip),up(x,[skip]),rtol=0,atol=0)

    def test_locked_recipe_mid_and_parameter_contract(self):
        base=yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        self.assertEqual(SCENARIOS,{'control':'NDC-Control','ex1':'NDC-Ex1'})
        for mode in SCENARIOS:
            cfg=config(mode);recipe=deepcopy(cfg);nested=recipe['backbone'].pop('nested',None);recipe['experiment']=base['experiment']
            self.assertEqual(recipe,base)
            net=get_backbone(cfg)
            self.assertEqual(sum(p.numel() for p in net.parameters()),189827 if mode=='control' else 470675)
            self.assertEqual(sum(isinstance(m,nn.MaxPool2d) for m in net.modules()),4)
            if mode=='ex1':
                self.assertEqual(nested,SETTINGS)
                for i,cin,mid,cout in [(1,256,128,64),(2,192,64,32),(3,128,32,16),(4,80,16,16)]:
                    c=getattr(net,f'up{i}').conv;self.assertEqual((c.in_channels,c.mid_channels,c.out_channels),(cin,mid,cout))
            for flag in ['fusion','attention','downsampling','deep_supervision']:
                wrong=deepcopy(cfg);wrong['backbone'][flag]={}
                with self.assertRaises(ValueError):get_backbone(wrong)
        for key,value in [('h_dropout',0.1),('h_init','kaiming_normal'),('extra_kernel_l2',True),('align_corners',False),('decoder_mid_channels',[128,96,64,40]),('resize','size'),('variant','partial')]:
            cfg=config('ex1');cfg['backbone']['nested'][key]=value
            with self.assertRaises(ValueError):get_backbone(cfg)

    def test_H_reference_structure_and_truncated_He_distribution(self):
        folder=ROOT/'tests/reference/ndc';spec=json.loads((folder/'sources.json').read_text())
        self.assertEqual(hashlib.sha256((folder/'helper_functions.py').read_bytes()).hexdigest(),spec['helper_sha256'])
        tree=ast.parse((folder/'helper_functions.py').read_text());standard=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='standard_unit')
        calls=[n for n in ast.walk(standard) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(sum(n.func.id=='Conv2D' for n in calls),2);self.assertEqual(sum(n.func.id=='Dropout' for n in calls),2)
        unit=HUnit(128,128)
        self.assertEqual([type(m) for m in unit],[nn.Conv2d,nn.ReLU,nn.Dropout,nn.Conv2d,nn.ReLU,nn.Dropout])
        normal_density=math.exp(-2)/(math.sqrt(2*math.pi));correction=math.sqrt(1-4*normal_density/math.erf(2/math.sqrt(2)))
        self.assertAlmostEqual(correction,KERAS_TRUNCATED_NORMAL_STD,places=14)
        for conv in [unit[0],unit[3]]:
            fan=conv.in_channels*9;target_var=2/fan;std_pre=math.sqrt(target_var)/correction
            self.assertEqual(conv.kernel_size,(3,3));self.assertEqual(conv.padding,(1,1));self.assertEqual(conv.groups,1)
            self.assertEqual(float(conv.bias.abs().sum()),0)
            self.assertLessEqual(float(conv.weight.abs().max()),2*std_pre+1e-7)
            self.assertLess(abs(float(conv.weight.var(unbiased=False))/target_var-1),0.04)
            self.assertLess(abs(float(conv.weight.mean())),math.sqrt(target_var)*0.03)
        self.assertFalse(any(isinstance(m,nn.modules.batchnorm._BatchNorm) for m in unit.modules()))
        unit.train();mask=unit[2](torch.ones(2,16,32,32));self.assertEqual(set(mask.unique().tolist()),{0.0,2.0})
        self.assertTrue(all((m==0).any() and (m==2).any() for m in mask[0]))
        unit.eval();x=torch.ones(1,128,4,4);torch.testing.assert_close(unit[2](x),x,rtol=0,atol=0)

    def test_ten_fusions_single_concat_order_node_reuse_and_backward(self):
        cfg=config('ex1');net=get_backbone(cfg).train();values={};inputs={};counts={}
        def output(name):
            def hook(m,a,o):values[name]=o;counts[name]=counts.get(name,0)+1
            return hook
        def input_hook(name):return lambda m,a:inputs.update({name:a[0]})
        for name,attr in [('x1','in_conv'),('x2','down1'),('x3','down2'),('x4','down3'),('x5','down4'),('D4','up1'),('D3','up2'),('D2','up3'),('D1','up4')]:
            getattr(net,attr).register_forward_hook(output(name))
            if name.startswith('D'):getattr(net,attr).conv.register_forward_pre_hook(input_hook(name))
        for name,mod in net.h_units.items():mod.register_forward_pre_hook(input_hook(name));mod.register_forward_hook(output(name))
        with patch('torch.cat',wraps=torch.cat) as concat:
            logits=net(torch.randn(2,3,33,35));self.assertEqual(concat.call_count,10)
        expected={'H01':('x2',['x1']),'H11':('x3',['x2']),'H21':('x4',['x3']),'D4':('x5',['x4']),
                  'H02':('H11',['x1','H01']),'H12':('H21',['x2','H11']),'D3':('D4',['x3','H21']),
                  'H03':('H12',['x1','H01','H02']),'D2':('D3',['x2','H11','H12']),'D1':('D2',['x1','H01','H02','H03'])}
        for name,(deeper,previous) in expected.items():
            reference=torch.cat([upsample_like(values[deeper],values[previous[0]]),*[values[n] for n in previous]],dim=1)
            torch.testing.assert_close(inputs[name],reference,rtol=0,atol=0)
            self.assertEqual(counts[name],1)
        self.assertEqual(logits.shape,(2,3,33,35))
        labels=torch.randint(0,3,(2,33,35));loss=get_criterion(cfg)(logits,labels,mode='train',mask_keep=labels!=255);loss.backward()
        for name,p in net.named_parameters():self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        optimizer=torch.optim.Adam(net.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay'])
        before={name:mod[0].weight.clone() for name,mod in net.h_units.items()};optimizer.step()
        for name,mod in net.h_units.items():self.assertFalse(torch.equal(before[name],mod[0].weight))

    def test_eval_and_separate_fusion_audit(self):
        net=get_backbone(config('ex1')).train();x=torch.randn(2,3,33,35);net(x).mean().backward();net.eval()
        rng=torch.get_rng_state().clone()
        with torch.inference_mode():a=net(x);b=net(x)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));torch.testing.assert_close(a,b,rtol=0,atol=0)
        # Fusion is a separate certification gate; main training/export is unfused.
        from scripts.audit_ndc_reparameterization import audit
        report=audit(net,sizes=((32,32),(33,35),(64,64)))
        self.assertEqual(report['H_nodes_preserved'],6)
        self.assertEqual((report['atol'],report['rtol']),(1e-4,1e-4))
        self.assertTrue(all(p['finite'] for p in report['probes']))
        Path('/tmp/ndc_initial_fusion_audit.json').write_text(json.dumps(report,indent=2)+'\n')

    def test_fusion_certification_rejects_changed_logits(self):
        from scripts.audit_ndc_reparameterization import audit
        class BoundedLogits(nn.Module):
            def __init__(self):
                super().__init__()
                self.logits=nn.Parameter(torch.tensor([1.0,2.0,3.0]).reshape(1,3,1,1),requires_grad=False)
                self.h_units=nn.ModuleDict({f'H{i}':nn.Identity() for i in range(6)})
            def forward(self,x):return self.logits.expand(x.shape[0],3,*x.shape[2:])
        class Changed(nn.Module):
            def __init__(self,m,offset):super().__init__();self.m=m;self.offset=offset
            @property
            def h_units(self):return self.m.h_units
            def forward(self,x):return self.m(x)+self.offset
        # Bound logits so a fixed perturbation cannot be hidden by rtol on a
        # randomly initialized NDC. Preserve H metadata to isolate logit checks.
        net=BoundedLogits().eval()
        for offset,expected_pass in [(0.0,True),(1e-5,True),(1.0,False)]:
            with self.subTest(offset=offset):
                with patch('scripts.audit_ndc_reparameterization.RPD_model_deploy',side_effect=lambda m,do_copy:Changed(m,offset)):
                    report=audit(net,sizes=((32,32),(33,35)))
                self.assertEqual((report['atol'],report['rtol']),(1e-4,1e-4))
                self.assertEqual(report['H_nodes_expected'],6)
                self.assertEqual(report['H_nodes_preserved'],6)
                self.assertEqual(report['status'],'passed' if expected_pass else 'not_certified')
                for probe in report['probes']:
                    self.assertTrue(probe['finite'])
                    self.assertEqual(probe['passed'],expected_pass)

    def test_export_metadata_and_original_objective(self):
        for mode in SCENARIOS:
            cfg=config(mode);net=get_backbone(cfg)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'weights.ckpt';payload={'state_dict':{'network.'+k:v for k,v in net.state_dict().items()},'nested':cfg['backbone'].get('nested'),'val_history':[]};torch.save(payload,path)
                report=export(cfg,path,Path(tmp)/'export',sizes=((32,32),(33,35)))
                self.assertEqual(report['nested_preserved'],cfg['backbone'].get('nested'))
                wrong=config('ex1' if mode=='control' else 'control')
                with self.assertRaises(ValueError):load_model(wrong,path)
                with self.assertRaises(ValueError):module(wrong).on_load_checkpoint(payload)
        original=subprocess.check_output(['git','show','d77d518:models/model_multimetrics.py'],cwd=ROOT,text=True);actual=(ROOT/'models/model_multimetrics.py').read_text()
        def body(text,name):return ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ['training_step','on_train_epoch_end','compute_loss','configure_optimizers','lr_scaling']:self.assertEqual(body(actual,name),body(original,name))

    def test_notebook_recipe_and_no_resource_fallback(self):
        nb=json.loads((ROOT/'notebook/kaggle_train_repdwnet_ndc.ipynb').read_text());ns={};exec(''.join(nb['cells'][1]['source']),ns)
        self.assertEqual(ns['EXPERIMENTS'],SCENARIOS);self.assertEqual(ns['GIT_REF'],'f/fuse-ndc');self.assertEqual(ns['DEVICES'],2)
        text='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('preflight_ndc_resources.py',text)
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(l for l in c['source'] if not l.lstrip().startswith('!')));self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])

if __name__=='__main__':
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--pipeline-smoke',action='store_true')
    p.add_argument('--config');p.add_argument('--output-dir');p.add_argument('--accelerator',default='cpu');p.add_argument('--devices',type=int,default=1);p.add_argument('--resume',action='store_true')
    args,rest=p.parse_known_args()
    if args.smoke or args.pipeline_smoke:
        if not args.config or not args.output_dir:p.error('config/output-dir required')
        cfg=yaml.safe_load(Path(args.config).read_text())
        if args.smoke:trainer_smoke(cfg,args.output_dir,args.accelerator,args.devices,args.resume)
        else:pipeline_smoke(cfg,args.output_dir,args.accelerator,args.devices)
    else:unittest.main(argv=[__file__,*rest])
