"""Paired single-GPU protocol: mathematical checks and persistent artifact checks."""
from copy import deepcopy
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from utils.tensorboard_compat import use_tensorboard_without_tensorflow
use_tensorboard_without_tensorflow()
from utils.ndc_colab import PairedSampler,PairedRankState,PROTOCOL,publish_backup,restore_backup,validate_recipe
from utils.ndc_campaign import config
from models import get_criterion
from utils.ndc_campaign import ROOT

class ColabTests(unittest.TestCase):
    def test_paired_sampling_counts_and_validation_identity(self):
        train=PairedSampler(range(1407));ids=list(train)
        self.assertEqual(len(ids),1408);self.assertEqual(len(set(ids)),1407);self.assertEqual(len(ids)//8,176)
        shuffled=torch.randperm(1407,generator=torch.Generator().manual_seed(1682409321)).tolist();shuffled+=[shuffled[0]]
        for step in range(176):
            self.assertEqual(ids[step*8:step*8+4],shuffled[step*8:step*8+8:2])
            self.assertEqual(ids[step*8+4:step*8+8],shuffled[step*8+1:step*8+8:2])
        train.set_epoch(1);self.assertNotEqual(ids,list(train))
        val=list(PairedSampler(range(772),shuffle=False));self.assertEqual(len(val),768);self.assertEqual(set(val),set(range(768)))
        self.assertEqual(val[:8],[0,2,4,6,1,3,5,7]);self.assertEqual(val,list(PairedSampler(range(772),shuffle=False)))

    def test_mean_gradient_adam_bn_and_dropout_match_two_replicas(self):
        torch.manual_seed(31)
        make=lambda:nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),nn.Dropout(.5),nn.Conv2d(4,3,1))
        actual=make();ref0=deepcopy(actual);ref1=deepcopy(actual);state=PairedRankState()
        torch.manual_seed(712);streams=[torch.get_rng_state().clone() for _ in range(2)]
        state.streams=[{'cpu':s.clone(),'cuda':None} for s in streams]
        adam=lambda m:torch.optim.Adam(m.parameters(),lr=2e-4,weight_decay=2e-4)
        opt=adam(actual);refopt=adam(ref0);criterion=get_criterion(config('ex1'))
        g=torch.Generator().manual_seed(714)
        for step in range(3):
            batches=[(torch.randn(4,3,8,8,generator=g),torch.randint(0,3,(4,8,8),generator=g)) for _ in range(2)]
            ref1.load_state_dict(ref0.state_dict());opt.zero_grad(set_to_none=True);refopt.zero_grad(set_to_none=True);ref1.zero_grad(set_to_none=True)
            for rank,(x,y) in enumerate(batches):
                reference=[ref0,ref1][rank]
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(streams[rank]);loss=criterion(reference(x),y,mode='train',mask_keep=y!=255);streams[rank]=torch.get_rng_state().clone();loss.backward()
                state.on_train_batch_start(None,SimpleNamespace(network=actual),None,rank)
                (criterion(actual(x),y,mode='train',mask_keep=y!=255)/2).backward()
                state.on_train_batch_end(None,SimpleNamespace(network=actual),None,None,rank)
            for p0,p1,p in zip(ref0.parameters(),ref1.parameters(),actual.parameters()):
                p0.grad=(p0.grad+p1.grad)/2;torch.testing.assert_close(p.grad,p0.grad,rtol=1e-5,atol=1e-6)
            opt.step();refopt.step()
            for a,b in zip(actual.state_dict().values(),ref0.state_dict().values()):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            self.assertEqual(int(actual[1].num_batches_tracked),step+1)
            for r in range(2):self.assertTrue(torch.equal(state.streams[r]['cpu'],streams[r]))
        resumed=PairedRankState();resumed.load_state_dict(state.state_dict());self.assertTrue(torch.equal(resumed.streams[0]['cpu'],streams[0]))

    def test_backup_restore_integrity_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir();(source/'source_commit.txt').write_text('a'*40+chr(10));(source/'last.ckpt').write_text('first')
            first=publish_backup(source,root/'drive');(source/'last.ckpt').write_text('second');second=publish_backup(source,root/'drive');self.assertNotEqual(first['slot'],second['slot'])
            meta=restore_backup(root/'drive',root/'restored');self.assertEqual(meta['protocol'],PROTOCOL);self.assertEqual((root/'restored/last.ckpt').read_text(),'second')
            (root/'drive'/meta['archive']).write_text('broken')
            with self.assertRaises(IOError):restore_backup(root/'drive',root/'corrupt')

    def test_locked_recipe_no_implicit_model_changes(self):
        cfg=config('ex1');cfg['train'].update(devices=1,accelerator='gpu',accumulate_grad_batches=2);cfg['execution']={'protocol':PROTOCOL,'backup_dir':'/tmp/backup'}
        self.assertEqual(validate_recipe(cfg),'ex1')
        for key,value in [('precision',16),('batch_size',8),('learning_rate',1e-3),('max_epoch',99)]:
            wrong=deepcopy(cfg);wrong['train'][key]=value
            with self.assertRaises(ValueError):validate_recipe(wrong)

    def test_notebook_syntax_defaults_and_resolved_configs(self):
        import ast,yaml
        notebook=json.loads((ROOT/'notebook/colab_train_repdwnet_ndc_a100.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                ast.parse(''.join(line for line in cell['source'] if not line.lstrip().startswith('!')))
                self.assertEqual(cell['outputs'],[]);self.assertIsNone(cell['execution_count'])
        ns={};exec(''.join(notebook['cells'][1]['source']),ns)
        self.assertEqual(ns['EXPERIMENT_MODES'],['control','ex1'])
        self.assertEqual(ns['DATASET_HANDLE'],'ndhoang2310/phenobench-dataset/versions/1')
        with tempfile.TemporaryDirectory() as tmp:
            ns.update(ROOT=Path(tmp),PERSISTENT=Path(tmp)/'drive',REPO_DIR=ROOT,
                      dataset_dir=Path('/dataset/PhenoBench'),backup_metas={},copy=__import__('copy'),
                      yaml=yaml,validate_recipe=validate_recipe,PROTOCOL=PROTOCOL)
            exec(''.join(notebook['cells'][5]['source']),ns)
            self.assertEqual(set(ns['RUNS']),{'control','ex1'})
            for mode,info in ns['RUNS'].items():self.assertEqual(validate_recipe(info['cfg']),mode)

if __name__=='__main__':torch.set_num_threads(2);unittest.main()
