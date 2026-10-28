"""CFF contracts, isolated ablations, P2 compatibility and artifact integration."""
import argparse,ast,json,subprocess,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
import yaml
from models import get_backbone,get_criterion
from models.rpdnet.fuse_cff import CascadeFeatureFusion,POLICIES,settings,upsample_like
from scripts.cff_smoke import module,trainer_smoke,pipeline_smoke
from scripts.export_cff import load_model,export
from scripts.profile_cff import profile
from utils.cff_campaign import ROOT,SCENARIOS,config

VARIANTS={'ex1':(2,'projection_bn'),'ex2':(1,'projection_bn'),'ex3':(3,'projection_bn'),'ex4':(2,'bn_only'),'ex5':(2,'identity')}

class CFFTests(unittest.TestCase):
    def test_p2_control_exact_and_common_host_initialization_rng(self):
        source=subprocess.check_output(['git','show','d77d518:models/rpdnet/RepDWNet.py'],cwd=ROOT,text=True)
        ns={};exec(compile(source,'P2_original','exec'),ns)
        torch.manual_seed(1682409321);original=ns['RepDWNet'](3).eval();rng=torch.get_rng_state().clone()
        for mode in SCENARIOS:
            torch.manual_seed(1682409321);net=get_backbone(config(mode)).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            for key,value in original.state_dict().items():
                if mode=='control' or key.startswith(('in_conv.','down','out_conv.')):
                    self.assertTrue(torch.equal(value,net.state_dict()[key]),key)
            if mode=='control':
                self.assertEqual(set(original.state_dict()),set(net.state_dict()))
                for h,w in [(32,32),(33,35)]:
                    x=torch.randn(1,3,h,w)
                    with torch.inference_mode():torch.testing.assert_close(original(x),net(x),rtol=0,atol=0)

    def test_matrix_recipe_fixed_mids_and_policy_lock(self):
        base=yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        self.assertEqual(len(SCENARIOS),6)
        for mode in SCENARIOS:
            cfg=config(mode);recipe=deepcopy(cfg);cff=recipe['backbone'].pop('cff',None);recipe['experiment']=base['experiment'];self.assertEqual(recipe,base)
            net=get_backbone(cfg)
            self.assertEqual(sum(isinstance(m,nn.MaxPool2d) for m in net.modules()),4)
            self.assertEqual(sum(isinstance(m,CascadeFeatureFusion) for m in net.modules()),0 if mode=='control' else 4)
            if mode!='control':
                self.assertEqual(cff,settings(*VARIANTS[mode]))
                for i,cin,mid,cout in [(1,128,128,64),(2,64,64,32),(3,32,32,16),(4,16,16,16)]:
                    block=getattr(net,f'up{i}').conv
                    self.assertEqual((block.in_channels,block.mid_channels,block.out_channels),(cin,mid,cout))
            for flag in ['fusion','attention','downsampling','deep_supervision','nested']:
                wrong=deepcopy(cfg);wrong['backbone'][flag]={}
                with self.assertRaises(ValueError):get_backbone(wrong)
        for key,value in [('dilation',4),('skip_mode','projection_only'),('conv_bias',True),('bn_eps',0.01),('resize','size'),('decoder_mid_channels',[64,32,16,8])]:
            cfg=config('ex1');cfg['backbone']['cff'][key]=value
            with self.assertRaises(ValueError):get_backbone(cfg)
        wrong=config('ex4');wrong['backbone']['cff']['dilation']=1
        with self.assertRaises(ValueError):get_backbone(wrong)

    def test_identical_common_weights_across_ablations(self):
        states={}
        for mode in VARIANTS:
            torch.manual_seed(1682409321);states[mode]=get_backbone(config(mode)).state_dict()
        baseline=states['ex1']
        for mode,state in states.items():
            for key,value in baseline.items():
                if '.fuse.skip.' not in key:self.assertTrue(torch.equal(value,state[key]),(mode,key))

    def test_operators_bn_init_resize_and_formula(self):
        for mode,(d,skip) in VARIANTS.items():
            torch.manual_seed(7);unit=CascadeFeatureFusion(64,settings(d,skip))
            conv=unit.deep[0]
            self.assertEqual((conv.kernel_size,conv.stride,conv.padding,conv.dilation,conv.groups),((3,3),(1,1),(d,d),(d,d),1));self.assertIsNone(conv.bias)
            self.assertLess(abs(float(conv.weight.detach().var(unbiased=False))/(2/(9*64))-1),0.05)
            for bn in [m for m in unit.modules() if isinstance(m,nn.BatchNorm2d)]:
                self.assertEqual((bn.eps,bn.momentum,bn.affine,bn.track_running_stats),(1e-5,0.1,True,True))
                self.assertTrue(torch.equal(bn.weight,torch.ones_like(bn.weight)));self.assertTrue(torch.equal(bn.bias,torch.zeros_like(bn.bias)))
            if skip=='projection_bn':self.assertIsInstance(unit.skip[0],nn.Conv2d);self.assertEqual(unit.skip[0].kernel_size,(1,1));self.assertIsNone(unit.skip[0].bias)
            elif skip=='bn_only':self.assertIsInstance(unit.skip,nn.BatchNorm2d)
            else:self.assertIsInstance(unit.skip,nn.Identity)
            for h,w in [(6,8),(7,9)]:
                unit.eval();deep=torch.randn(2,64,3,4);shallow=torch.randn(2,64,h,w)
                up=torch.nn.functional.interpolate(deep,scale_factor=2,mode='bilinear',align_corners=True)
                dy,dx=h-up.shape[-2],w-up.shape[-1];expected=torch.nn.functional.pad(up,[dx//2,dx-dx//2,dy//2,dy-dy//2])
                torch.testing.assert_close(upsample_like(deep,shallow),expected,rtol=0,atol=0)
                with torch.inference_mode(),patch('torch.cat',wraps=torch.cat) as cat:
                    actual=unit(deep,shallow);reference=torch.relu(unit.deep(expected)+unit.skip(shallow))
                self.assertEqual(cat.call_count,0);torch.testing.assert_close(actual,reference,rtol=0,atol=0)
            unit.train();unit(torch.randn(2,64,3,4),torch.randn(2,64,6,8));self.assertEqual(int(unit.deep[1].num_batches_tracked),1)
            unit.eval();unit(torch.randn(2,64,3,4),torch.randn(2,64,6,8));self.assertEqual(int(unit.deep[1].num_batches_tracked),1)

    def test_full_graph_backward_optimizer_and_four_fusions(self):
        for mode in SCENARIOS:
            cfg=config(mode);net=get_backbone(cfg).train();calls=[]
            for m in net.modules():
                if isinstance(m,CascadeFeatureFusion):m.register_forward_hook(lambda m,a,o:calls.append(o.shape[1]))
            x=torch.randn(2,3,33,35);labels=torch.randint(0,3,(2,33,35))
            out=net(x);self.assertEqual(out.shape,(2,3,33,35));self.assertEqual(calls,[] if mode=='control' else [128,64,32,16])
            loss=get_criterion(cfg)(out,labels,mode='train',mask_keep=labels!=255);loss.backward();self.assertTrue(torch.isfinite(loss))
            for name,p in net.named_parameters():self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            before={k:p.detach().clone() for k,p in net.named_parameters() if '.fuse.' in k}
            opt=torch.optim.Adam(net.parameters(),lr=cfg['train']['learning_rate'],weight_decay=cfg['train']['weight_decay']);opt.step()
            for k,p in net.named_parameters():
                if k in before:self.assertFalse(torch.equal(before[k],p),k)

    def test_strict_checkpoint_export_crossvariant_and_resume_metadata(self):
        for mode in SCENARIOS:
            cfg=config(mode);net=get_backbone(cfg)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'weights.ckpt';payload={'state_dict':{'network.'+k:v for k,v in net.state_dict().items()},'cff':cfg['backbone'].get('cff'),'val_history':[]};torch.save(payload,path)
                result=export(cfg,path,Path(tmp)/'out',sizes=((32,32),(33,35)))
                self.assertEqual(result['cff_preserved'],cfg['backbone'].get('cff'))
                self.assertTrue(all(p['max_abs_error']==0 for p in result['parity']))
                for other in SCENARIOS:
                    if other==mode:continue
                    with self.assertRaises(ValueError):load_model(config(other),path)
                    with self.assertRaises(ValueError):module(config(other)).on_load_checkpoint(payload)

    def test_cost_dilation_invariance_and_skip_ablation(self):
        records={mode:profile(mode,sizes=((32,32),)) for mode in SCENARIOS}
        self.assertEqual(records['control']['trainable_params'],189827)
        self.assertEqual(records['control']['conv_macs'],13447987200//576)
        self.assertEqual(records['ex1']['trainable_params'],375347)
        self.assertEqual(records['ex4']['trainable_params'],353587)
        self.assertEqual(records['ex5']['trainable_params'],353107)
        for mode in ['ex2','ex3']:
            for field in ['trainable_params','conv_macs']:self.assertEqual(records[mode][field],records['ex1'][field])
        self.assertGreater(records['ex1']['conv_macs'],records['ex4']['conv_macs']);self.assertEqual(records['ex4']['conv_macs'],records['ex5']['conv_macs'])

    def test_original_objective_and_notebook_contract(self):
        original=subprocess.check_output(['git','show','d77d518:models/model_multimetrics.py'],cwd=ROOT,text=True);actual=(ROOT/'models/model_multimetrics.py').read_text()
        def body(text,name):return ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ['training_step','on_train_epoch_end','compute_loss','configure_optimizers','lr_scaling']:self.assertEqual(body(actual,name),body(original,name))
        nb=json.loads((ROOT/'notebook/kaggle_train_repdwnet_cff.ipynb').read_text());ns={};exec(''.join(nb['cells'][1]['source']),ns)
        self.assertEqual(ns['EXPERIMENTS'],SCENARIOS);self.assertEqual(ns['GIT_REF'],'f/fuse-cff');self.assertEqual(ns['DEVICES'],2)
        for c in nb['cells']:
            if c['cell_type']=='code':
                ast.parse(''.join(l for l in c['source'] if not l.lstrip().startswith('!')));self.assertEqual(c['outputs'],[]);self.assertIsNone(c['execution_count'])
        text='\n'.join(''.join(c['source']) for c in nb['cells'])
        self.assertNotIn('NDC',text);self.assertNotIn('keras',text);self.assertIn('preflight_cff_resources.py',text)
        self.assertIn('PREFLIGHT_PASSED',text)
        ref=json.loads((ROOT/'tests/reference/cff/sources.json').read_text());self.assertEqual(ref['commit'],'dd2bf78ff3f9ff05f6d35c40a3ee216158d9e892')

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
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    else:unittest.main(argv=[__file__]+rest)
