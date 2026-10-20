"""PhuongTU — execute notebook setup/logger/gates; never train or run GPU probes."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(remote, output):
    reports, pins = [], set()
    with tempfile.TemporaryDirectory(prefix='lsh-notebooks-') as temporary:
        fixture = Path(temporary)
        dataset = fixture / 'dataset'
        for relative in ('train/images', 'train/semantics', 'val/images', 'val/semantics', 'test/images'):
            (dataset / relative).mkdir(parents=True)
        notebooks = sorted((ROOT / 'notebook').glob('kaggle_train_p3_lsh_*.ipynb'))
        assert len(notebooks) == 2
        for notebook in notebooks:
            nb = json.loads(notebook.read_text())
            assert nb['metadata']['authors'] == [{'name': 'PhuongTU'}]
            def source(i):
                cell = nb['cells'][i]
                return ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    compile(source(i), f'{notebook.name}:{i}', 'exec')
                    assert cell['execution_count'] is None and cell['outputs'] == []
                    assert 'class RepDWNet' not in source(i)
            setup = source(4)
            assert 'GITHUB_BRANCH = "f/lsh"' in setup
            pin = re.search(r'SOURCE_COMMIT = "([0-9a-f]{40})"', setup).group(1)
            pins.add(pin)
            setup = setup.replace('"/kaggle/working"', repr(str(fixture / 'working')))
            setup = setup.replace('DATASET_DIR = ""', f'DATASET_DIR = {str(dataset)!r}')
            if not remote:
                setup = setup.replace('GITHUB_REPO = "https://github.com/FUAVX/RPD_Reproduce.git"',
                                      f'GITHUB_REPO = {str(ROOT)!r}')
            namespace = dict(subprocess=subprocess, sys=sys)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(setup, notebook.name, 'exec'), namespace)
            launch = namespace['launch']
            assert launch.manifest['source_commit'] == pin
            assert launch.manifest['source_branch'] == 'f/lsh'
            assert launch.manifest['source_diff'] == launch.manifest['source_status'] == ''
            assert launch.manifest['active_config']['train']['devices'] == 2
            assert launch.manifest['effective_batch'] == 8
            assert launch.manifest['resume_checkpoint'] is None
            assert not (launch.repo / 'results').exists()
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source(6), namespace)  # Real logger cell and subprocess event IO.
            assert launch.manifest['logger_passed']
            import torch
            gpu_gate = 'not executed; GPU runtime present, empty dataset fixture'
            if not torch.cuda.is_available():
                try:
                    exec(source(8), namespace)
                except RuntimeError as error:
                    assert 'visible GPU' in str(error)
                else:
                    raise AssertionError('CPU unexpectedly passed GPU preflight')
                gpu_gate = 'blocked as expected on CPU; no GPU pass claimed'
            assert not launch.manifest['preflight_passed']
            try:
                exec(source(10), namespace)
            except RuntimeError as error:
                assert 'must pass before train' in str(error)
            else:
                raise AssertionError('Training was not gated')
            with contextlib.redirect_stdout(io.StringIO()):
                exec(source(12), namespace)
            with zipfile.ZipFile(namespace['artifact_path']) as archive:
                assert 'run_manifest.json' in archive.namelist()
                assert not any(name.startswith('source/') for name in archive.namelist())
            reports.append(dict(notebook=notebook.name, source_commit=pin, passed=True,
                                setup_real_git_clone='GitHub remote' if remote else 'local Git remote',
                                logger_event_io='passed', gpu_preflight_gate=gpu_gate,
                                training_gate='blocked as expected', artifact_packaging='passed'))
    assert len(pins) == 1
    Path(output).write_text(json.dumps(dict(author='PhuongTU', reports=reports,
        schema_validation='nbformat checked separately in base Python',
        actual_phenobench_gpu_or_training_executed=False), indent=2) + '\n')
    print('Both LSH notebook checks passed:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--output', default='/tmp/lsh_notebook_checks.json')
    args = parser.parse_args()
    main(args.remote, args.output)
