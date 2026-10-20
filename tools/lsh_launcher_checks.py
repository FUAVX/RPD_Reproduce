"""PhuongTU — launcher integration checks; no GPU/data training is claimed."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import lsh_launcher as launcher

ROOT = Path(__file__).resolve().parents[1]


def expect(exception, message, call):
    try:
        call()
    except exception as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError(f'Expected {exception.__name__}: {message}')


def main(output):
    checks = []
    with tempfile.TemporaryDirectory(prefix='lsh-launcher-') as temporary:
        fixture = Path(temporary)
        repo = fixture / 'repo'
        repo.mkdir()
        files = ['tools/lsh_runtime/sitecustomize.py', 'tools/lsh_logger_smoke.py',
                 'configs/p3_scenarios/P3_LSH_L1_WDice.yaml', 'configs/p3_scenarios/P3_LSH_L2_WDice.yaml']
        for name in files:
            destination = repo / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / name).read_bytes())
        (repo / '.gitignore').write_text('__pycache__/\n*.pyc\n')
        def git(*arguments):
            return subprocess.check_output(['git', *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT)
        git('init', '-b', 'f/lsh')
        git('add', '.')
        git('-c', 'user.name=PhuongTU', '-c', 'user.email=lsh-fixture@example.invalid', 'commit', '-m', 'LSH fixture')
        pin = git('rev-parse', 'HEAD').strip()
        dataset = fixture / 'input/phenobench'
        for relative in ('train/images', 'train/semantics', 'val/images', 'val/semantics', 'test/images'):
            (dataset / relative).mkdir(parents=True)
        first = None
        for level in (1, 2):
            config = f'configs/p3_scenarios/P3_LSH_L{level}_WDice.yaml'
            def prepare(token, **kwargs):
                return launcher.LSHRun(repo, fixture / token, pin, config, input_root=fixture / 'input', **kwargs)
            with contextlib.redirect_stdout(io.StringIO()):
                current = prepare(f'run_l{level}')
            assert current.manifest['effective_batch'] == 8
            assert current.manifest['checkpoint_rule']['monitor'] == 'val_weed_iou'
            assert current.manifest['resume_checkpoint'] is None
            assert current.cfg['train']['devices'] == 2
            env = current.environment()
            assert env['P3_TENSORBOARD_NO_TF'] == '1'
            assert env['PYTHONPATH'].split(':')[:2] == [str(current.hook_dir), str(repo)]
            current.verify()
            original = current.active_config.read_bytes()
            current.active_config.write_bytes(original + b'\n# altered\n')
            expect(RuntimeError, 'Active config changed', current.verify)
            current.active_config.write_bytes(original)
            hook = current.hook.read_bytes()
            current.hook.write_bytes(hook + b'\n# altered\n')
            expect(RuntimeError, 'Runtime hook changed', current.verify)
            current.hook.write_bytes(hook)
            source = repo / config
            original_source = source.read_bytes()
            source.write_bytes(original_source + b'\n# dirty\n')
            expect(RuntimeError, 'Source must be clean', current.verify)
            source.write_bytes(original_source)
            expect(RuntimeError, 'must pass before train', current.train)
            # Exercise the actual subprocess helper, copied runtime hook and event IO.
            with contextlib.redirect_stdout(io.StringIO()):
                current.logger_smoke()
            assert current.manifest['logger_passed']
            import torch
            if not torch.cuda.is_available():
                expect(RuntimeError, 'visible GPU', current.preflight)
                assert not current.manifest['preflight_passed']
            expect(RuntimeError, 'must pass before train', current.train)
            checkpoint = current.directory / 'training_output/checkpoints/last.ckpt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'test fixture; never loaded by torch or training')
            excluded = current.directory / 'source/excluded.txt'
            excluded.parent.mkdir()
            excluded.write_text('fixture')
            with contextlib.redirect_stdout(io.StringIO()):
                artifact = current.package()
            with zipfile.ZipFile(artifact) as archive:
                assert 'run_manifest.json' in archive.namelist()
                assert 'training_output/checkpoints/last.ckpt' in archive.namelist()
                assert 'source/excluded.txt' not in archive.namelist()
            resume = dict(resume_checkpoint=str(checkpoint), resume_config=str(current.active_config),
                          resume_manifest=str(current.directory / 'run_manifest.json'))
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = prepare(f'resumed_l{level}', **resume)
            assert resumed.manifest['resume_checkpoint_sha256'] == launcher.sha256(checkpoint)
            expect(ValueError, 'Resume needs', lambda: prepare(f'incomplete_l{level}', resume_checkpoint=str(checkpoint)))
            previous_manifest = json.loads(Path(resume['resume_manifest']).read_text())
            wrong_manifest = fixture / f'wrong_l{level}.json'
            for key, value, message in [('source_commit', '0' * 40, 'scenario/source mismatch'),
                                        ('scenario', 'OtherScenario', 'scenario/source mismatch'),
                                        ('config_sha256', 'invalid', 'active config mismatch')]:
                wrong = dict(previous_manifest, **{key: value})
                wrong_manifest.write_text(json.dumps(wrong))
                altered = dict(resume, resume_manifest=str(wrong_manifest))
                expect(ValueError, message, lambda: prepare(f'wrong_{key}_l{level}', **altered))
            checkpoint.write_bytes(b'modified fixture')
            expect(ValueError, 'checkpoint hash mismatch', lambda: prepare(f'wrong_checkpoint_l{level}', **resume))
            # Check fresh/resume command and artifact path without dispatching training.
            checkpoint.write_bytes(b'test fixture; never loaded by torch or training')
            current.manifest['preflight_passed'] = True
            current.manifest['preflight_report_hashes'] = {}
            calls = []
            real_run = launcher.run
            launcher.run = lambda command, **kwargs: calls.append((command, kwargs))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    current.train()
                assert '--resume' not in calls[-1][0]
                resumed.manifest.update(logger_passed=True, preflight_passed=True,
                    logger_report_sha256=current.manifest['logger_report_sha256'], preflight_report_hashes={})
                report = resumed.directory / 'logger_smoke/report.json'
                report.parent.mkdir()
                report.write_bytes((current.directory / 'logger_smoke/report.json').read_bytes())
                with contextlib.redirect_stdout(io.StringIO()):
                    resumed.train()
                assert calls[-1][0][-3:] == ['--ckpt_path', str(checkpoint), '--resume']
                assert calls[-1][1]['env']['P3_TENSORBOARD_NO_TF'] == '1'
                assert calls[-1][1]['env']['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] == '1'
            finally:
                launcher.run = real_run
            checks.append(dict(model=f'L{level}', passed=True,
                checks='prepare/provenance/config/hook/source guards, real logger IO, CPU GPU-gate failure, resume hashes/scenario, mocked train CLI and artifact packaging'))
    Path(output).write_text(json.dumps(dict(author='PhuongTU', checks=checks,
        real_training_or_real_dataset_executed=False, real_logger_executed=True,
        training_process_mocked=True, fixture_source='temporary minimal Git repository'), indent=2) + '\n')
    print('LSH launcher checks passed:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/lsh_launcher_checks.json')
    main(parser.parse_args().output)
