"""AAA evaluator guards: absent/ambiguous artifacts and corrupted fixed filters."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
import aaa_evaluation as aaa
import dpp_evaluation as common


class AAAEvaluationTests(unittest.TestCase):
    def test_missing_trained_artifacts_fail_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):aaa.discover_artifacts(root)
            base=root/'model/Loss_Models/pre-model';base.mkdir(parents=True)
            (base/'P2_Ex7_RepDWNet_WDice-Loss_artifacts.zip').touch()
            with self.assertRaises(FileNotFoundError):aaa.discover_artifacts(root)

    def test_ambiguous_runs_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'model/Loss_Models/pre-model';base.mkdir(parents=True)
            (base/'P2_Ex7_RepDWNet_WDice-Loss_artifacts.zip').touch()
            models=root/'model/Down_Models/AAA';models.mkdir(parents=True)
            for run in [1,2]:(models/f'P3_AAA_Ex1_Early_aaa_tri3_run{run}_artifacts.zip').touch()
            with self.assertRaises(FileNotFoundError):aaa.discover_artifacts(root)

    def test_checkpoint_filter_audit_rejects_changed_fixed_coefficients(self):
        repo = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo))
        from models.rpdnet.aaa import tri3_filter
        state={'network.down1.0.filt':tri3_filter(16),'network.down2.0.filt':tri3_filter(32)}
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / 'RPD_P2').symlink_to(repo, target_is_directory=True)
            path=Path(tmp)/'artifact.zip'
            entry={'artifact':str(path),'member':'selected.ckpt','name':'AAA-Early',
                   'selection':'best_mIoU','stages':[1,2]}
            def write():
                buffer=io.BytesIO();torch.save({'state_dict':state},buffer)
                with zipfile.ZipFile(path,'w') as z:z.writestr('selected.ckpt',buffer.getvalue())
            write()
            self.assertTrue(aaa.filter_analysis(workspace,[entry],Path(tmp)/'out').fixed_filter_matches.all())
            state['network.down2.0.filt'].zero_();write()
            with self.assertRaises(ValueError):aaa.filter_analysis(workspace,[entry],Path(tmp)/'out')


if __name__=='__main__':unittest.main()
