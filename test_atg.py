"""ATG parity with official source, host identity, topology, gradients and pipeline."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
import torch
from torch import nn
import torch.nn.init as init
import yaml
from models import get_backbone,get_criterion
from models.rpdnet.fuse_atg import GridAttentionBlock2D,AttentionSkip,official_init
from utils.atg_campaign import ROOT,SCENARIOS,config
from scripts.atg_smoke import module,trainer_smoke,pipeline_smoke
from scripts.export_atg import export,load_model


def official_namespace():
    folder=ROOT/'tests/reference/atg'
    tree=ast.parse((folder/'models/networks_other.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['weights_init_kaiming','init_weights']]
    ns={'init':init,'__name__':'official_atg_fixture'}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_initialization','exec'),ns)
    tree=ast.parse((folder/'models/layers/grid_attention_layer.py').read_text())
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='models.networks_other')]
    exec(compile(tree,'official_gate','exec'),ns)
    # Faithful 3D->2D adaptation of the CT-Single combine wrapper only.
    text=(folder/'models/networks/unet_CT_single_att_dsv_3D.py').read_text().replace('Conv3d','Conv2d').replace('BatchNorm3d','BatchNorm2d').replace('GridAttentionBlock3D','GridAttentionBlock2D')
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MultiAttentionBlock')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'official_CT_single_wrapper_2D','exec'),ns)
    return ns


class ATGTests(unittest.TestCase):
    def test_exact_control_and_all_common_host_weights_and_rng(self):
        text=subprocess.check_output(['git','show','d77d518:models/rpdnet/RepDWNet.py'],cwd=ROOT,text=True)
        namespace={};exec(compile(text,'P2_original','exec'),namespace)
        torch.manual_seed(1682409321);original=namespace['RepDWNet'](3).eval();rng=torch.get_rng_state().clone()
        for mode in SCENARIOS:
            torch.manual_seed(1682409321);model=get_backbone(config(mode)).eval()
            self.assertTrue(torch.equal(rng,torch.get_rng_state()),mode)
            current=model.state_dict()
            for name,value in original.state_dict().items():self.assertTrue(torch.equal(value,current[name]),(mode,name))
            self.assertEqual(sum(isinstance(m,nn.MaxPool2d) for m in model.modules()),4)
            self.assertFalse(any('aux_head' in k or k.startswith('fpn.') for k in current))
            if mode=='control':
                self.assertEqual(set(current),set(original.state_dict()))
                for h,w in [(32,32),(33,35)]:
                    x=torch.randn(1,3,h,w)
                    with torch.inference_mode():torch.testing.assert_close(model(x),original(x),rtol=0,atol=0)

    def test_recipe_and_experiment_mapping(self):
        baseline=yaml.safe_load((ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml').read_text())
        for mode in SCENARIOS:
            cfg=config(mode);recipe=deepcopy(cfg);settings=recipe['backbone'].pop('attention',None);recipe['experiment']=baseline['experiment']
            self.assertEqual(recipe,baseline)
            if mode!='control':
                i=int(mode[2:]);self.assertEqual(cfg['experiment']['id'],f'ATG-Ex{i}')
                self.assertEqual(settings['strategy'],'PRG' if i in [1,2,5,6] else 'FBG')
                self.assertEqual(settings['block'],'GenStan' if i%2 else 'GenCom')
                self.assertEqual(settings['inter_ratio'],0.5 if i<=4 else 1.0)
            for flag in ['fusion','downsampling','deep_supervision']:
                wrong=deepcopy(cfg);wrong['backbone'][flag]={}
                with self.assertRaises(ValueError):get_backbone(wrong)

    def test_official_generic_and_combine_initialization_output_gradient_parity(self):
        ns=official_namespace()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for cx,cg,inter in [(128,128,64),(64,64,32),(32,128,32)]:
                for combine in [False,True]:
                    torch.manual_seed(100)
                    ref=ns['MultiAttentionBlock'](cx,cg,inter,'concatenation',(2,2)) if combine else ns['GridAttentionBlock2D'](cx,cg,inter,sub_sample_factor=(2,2))
                    torch.manual_seed(100)
                    actual=AttentionSkip(cx,cg,inter,True) if combine else GridAttentionBlock2D(cx,cg,inter)
                    refs=ref.state_dict();current=actual.state_dict()
                    if combine:refs={k.replace('gate_block_1.','gate.'):v for k,v in refs.items()}
                    self.assertEqual(set(refs),set(current))
                    for k,v in refs.items():self.assertTrue(torch.equal(v,current[k]),k)
                    x=torch.randn(2,cx,9,11,requires_grad=True);g=torch.randn(2,cg,3,4,requires_grad=True)
                    xr=x.detach().clone().requires_grad_();gr=g.detach().clone().requires_grad_()
                    a,alpha=actual(x,g);b,beta=ref(xr,gr)
                    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(alpha,beta,rtol=0,atol=0)
                    a.square().mean().backward();b.square().mean().backward()
                    torch.testing.assert_close(x.grad,xr.grad,rtol=0,atol=0);torch.testing.assert_close(g.grad,gr.grad,rtol=0,atol=0)
                    for (_,pa),(_,pb) in zip(actual.named_parameters(),ref.named_parameters()):torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_gating_sources_concat_order_and_all_gradients(self):
        for mode in [m for m in SCENARIOS if m!='control']:
            cfg=config(mode);net=get_backbone(cfg).train();seen={};calls={'gating':0}
            net.attention_gating.register_forward_hook(lambda m,a,o: (calls.update(gating=calls['gating']+1),seen.update(gating=o)) and None)
            net.up1.register_forward_hook(lambda m,a,o:seen.update(up1=o))
            net.up2.register_forward_hook(lambda m,a,o:seen.update(up2=o))
            for level in ['4','3','2']:
                net.attention_gates[level].register_forward_pre_hook(lambda m,a,level=level:seen.update({f'g{level}':a[1]}))
            out=net(torch.randn(2,3,33,35),return_attention=True)
            self.assertEqual(calls['gating'],1);self.assertIs(seen['g4'],seen['gating'])
            prg=cfg['backbone']['attention']['strategy']=='PRG'
            self.assertIs(seen['g3'],seen['up1'] if prg else seen['gating']);self.assertIs(seen['g2'],seen['up2'] if prg else seen['gating'])
            for level,cx,shape in [('4',128,(4,4)),('3',64,(8,8)),('2',32,(16,17))]:
                block=net.attention_gates[level];gate=block.gate
                self.assertEqual(gate.theta.in_channels,cx);self.assertIsNone(gate.theta.bias)
                self.assertEqual(gate.theta.out_channels,int(cx*cfg['backbone']['attention']['inter_ratio']))
                for c in [gate.phi,gate.psi,gate.W[0]]:self.assertIsNotNone(c.bias)
                self.assertEqual(block.combine_gates is not None,cfg['backbone']['attention']['block']=='GenCom')
                alpha=out['attention'][f'ag{level}'];self.assertEqual(alpha.shape,(2,1,*shape))
                self.assertTrue(torch.isfinite(alpha).all() and alpha.min()>=0 and alpha.max()<=1)
            target=torch.randint(0,3,(2,33,35));loss=get_criterion(cfg)(out['logits'],target,mode='train',mask_keep=target!=255);loss.backward()
            for name,p in net.named_parameters():self.assertIsNotNone(p.grad,(mode,name));self.assertTrue(torch.isfinite(p.grad).all(),(mode,name))
            probe=module(cfg);probe._trainer=SimpleNamespace(max_epochs=100,current_epoch=0)
            optimizer=probe.configure_optimizers()[0][0]
            self.assertEqual(optimizer.defaults['lr'],0.0002);self.assertEqual(optimizer.defaults['weight_decay'],0.0002)
            for block in [net.up1,net.up2,net.up3,net.up4]:self.assertTrue(block.up.align_corners)

    def test_strategy_parameter_deltas_and_unchanged_training(self):
        counts={mode:sum(p.numel() for p in get_backbone(config(mode)).parameters()) for mode in SCENARIOS}
        for prg,fbg,delta in [('ex1','ex3',3584),('ex2','ex4',3584),('ex5','ex7',7168),('ex6','ex8',7168)]:self.assertEqual(counts[fbg]-counts[prg],delta)
        original=subprocess.check_output(['git','show','d77d518:models/model_multimetrics.py'],cwd=ROOT,text=True)
        actual=(ROOT/'models/model_multimetrics.py').read_text()
        def body(source,name):return ast.dump(next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ['training_step','on_train_epoch_end','compute_loss','configure_optimizers','lr_scaling']:self.assertEqual(body(actual,name),body(original,name))

    def test_export_and_metadata_guards(self):
        for mode in SCENARIOS:
            cfg=config(mode);net=get_backbone(cfg)
            with tempfile.TemporaryDirectory() as tmp:
                ckpt={'state_dict':{'network.'+k:v for k,v in net.state_dict().items()},'attention':cfg['backbone'].get('attention'),'val_history':[]}
                path=Path(tmp)/'model.ckpt';torch.save(ckpt,path)
                report=export(cfg,path,Path(tmp)/'export',sizes=((32,32),(33,35)))
                self.assertEqual(report['attention_preserved'],cfg['backbone'].get('attention'))
                wrong=deepcopy(cfg)
                wrong['backbone']['attention']=deepcopy(config('ex1' if mode!='ex1' else 'ex2')['backbone']['attention'])
                with self.assertRaises(ValueError):load_model(wrong,path)
                with self.assertRaises(ValueError):module(wrong).on_load_checkpoint(ckpt)
            if mode!='control':
                for key,val in [('align_corners',True),('sub_sample_factor',[1,1]),('gating_channels',64),('inter_ratio',0.25),('mode','softmax')]:
                    wrong=deepcopy(cfg);wrong['backbone']['attention'][key]=val
                    with self.assertRaises(ValueError):get_backbone(wrong)

    def test_real_optimizer_updates_and_bn(self):
        cfg=config('ex2');net=module(cfg).train();net._trainer=SimpleNamespace(max_epochs=100,current_epoch=0)
        optim=net.configure_optimizers()[0][0]
        all_ids={id(p) for group in optim.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in all_ids for p in net.network.attention_gates.parameters()))
        x=torch.randn(2,3,32,32);y=torch.randint(0,3,(2,32,32));before={k:v.clone() for k,v in net.network.state_dict().items()}
        loss=net.compute_loss(net(x),y,'train',mask_keep=y!=255);loss.backward();optim.step()
        for prefix in ['attention_gating.','attention_gates.4.','attention_gates.3.','attention_gates.2.']:
            self.assertTrue(any(not torch.equal(before[k],v) for k,v in net.network.state_dict().items() if k.startswith(prefix) and v.is_floating_point()))
        for mod in net.network.modules():
            if isinstance(mod,nn.BatchNorm2d):self.assertEqual(mod.eps,1e-5);self.assertEqual(mod.momentum,0.1)
        net.eval()
        with torch.no_grad():a=net(x);b=net(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_notebook_modes_and_syntax(self):
        nb=json.loads((ROOT/'notebook/kaggle_train_repdwnet_atg.ipynb').read_text());namespace={}
        exec(''.join(nb['cells'][1]['source']),namespace)
        self.assertEqual(namespace['EXPERIMENTS'],SCENARIOS);self.assertEqual(namespace['GIT_REF'],'f/fuse-atg');self.assertEqual(namespace['DEVICES'],2)
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
