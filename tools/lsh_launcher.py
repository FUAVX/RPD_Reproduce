"""PhuongTU — shared GitHub-pinned launcher for L0/L1/L2; never auto-trains."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import zipfile

import yaml

CHECKPOINT_RULE = dict(monitor='val_weed_iou', mode='max',
                       selection='best checkpoint from the existing val_weed_iou callback; same rule for L0/L1/L2')
PACKAGES = ('torch', 'torchvision', 'pytorch-lightning', 'torchmetrics', 'tensorboard',
            'numpy', 'timm', 'setuptools', 'pandas', 'scikit-image', 'PyYAML')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def valid_dataset(path):
    return all((path / split / kind).is_dir() for split in ('train', 'val')
               for kind in ('images', 'semantics')) and (path / 'test/images').is_dir()


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def run(command, cwd=None, env=None, log_path=None):
    print('+', ' '.join(map(str, command)), flush=True)
    with Path(log_path).open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            status = process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                process.wait()
            raise
        finally:
            process.stdout.close()
        if status:
            raise subprocess.CalledProcessError(status, command)


class LSHRun:
    def __init__(self, repo_dir, run_dir, source_commit, config_relative, num_devices=2,
                 num_workers=2, dataset_dir='', input_root='/kaggle/input',
                 resume_checkpoint='', resume_config='', resume_manifest=''):
        self.repo = Path(repo_dir).resolve()
        self.directory = Path(run_dir).resolve()
        self.source_commit = source_commit
        self.config_relative = config_relative
        self.num_devices = num_devices
        self.resume_checkpoint = resume_checkpoint
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / 'run_manifest.json').exists():
            raise RuntimeError('Use a fresh run token; an existing manifest must not be overwritten')
        if not isinstance(num_devices, int) or num_devices < 1 or num_workers < 0:
            raise ValueError('GPU count must be positive and worker count nonnegative')
        commit, diff, status = self.source_state()
        source_config = self.repo / config_relative
        self.cfg = yaml.safe_load(source_config.read_text())
        if self.cfg['backbone']['name'] not in ('RepDWNet', 'RepDWNetLSHL1', 'RepDWNetLSHL2'):
            raise ValueError('LSH launcher accepts B1 L0, L1 or L2; no SRR combination')
        if dataset_dir:
            dataset = Path(dataset_dir).resolve()
        else:
            candidates = sorted({p.parent.parent.resolve() for p in Path(input_root).rglob('images')
                                 if p.parent.name == 'train' and valid_dataset(p.parent.parent)})
            if len(candidates) != 1:
                raise RuntimeError('Add Input PhenoBench or set DATASET_DIR; expected exactly one valid root')
            dataset = candidates[0]
        if not valid_dataset(dataset):
            raise RuntimeError('Dataset needs train/val images+semantics and test/images')
        self.cfg['data'].update(path_to_dataset=str(dataset), num_workers=num_workers)
        self.cfg['train']['devices'] = num_devices
        self.cfg['val']['devices'] = 1
        self.active_config = self.directory / 'active_config.yaml'
        self.active_config.write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        self.hook_dir = self.directory / 'python_runtime_compat'
        self.hook_dir.mkdir()
        self.hook = self.hook_dir / 'sitecustomize.py'
        self.hook.write_bytes((self.repo / 'tools/lsh_runtime/sitecustomize.py').read_bytes())
        self.manifest = dict(author='PhuongTU', scenario=self.cfg['experiment']['id'],
            run_token=self.directory.name, source_repository='https://github.com/FUAVX/RPD_Reproduce.git',
            source_branch='f/lsh', requested_commit=source_commit, source_commit=commit,
            source_diff=diff, source_status=status, source_config=config_relative,
            source_config_sha256=sha256(source_config), config_sha256=sha256(self.active_config),
            active_config=self.cfg, dataset_path=str(dataset), num_devices=num_devices,
            per_device_batch=self.cfg['train']['batch_size'],
            effective_batch=num_devices * self.cfg['train']['batch_size'] * self.cfg['train']['accumulate_grad_batches'],
            checkpoint_rule=CHECKPOINT_RULE, versions=versions(), python=sys.version,
            tensorboard_runtime_fix=dict(startup_sha256=sha256(self.hook),
                hook_directory=str(self.hook_dir), P3_TENSORBOARD_NO_TF='1',
                source='handoff notf startup hook; inherited through PYTHONPATH'),
            resume_checkpoint=resume_checkpoint or None, resume_config=resume_config or None,
            resume_manifest=resume_manifest or None, status='prepared',
            logger_passed=False, preflight_passed=False)
        self.validate_resume(resume_checkpoint, resume_config, resume_manifest)
        (self.directory / 'source_diff.patch').write_text(diff)
        (self.directory / 'source_status.txt').write_text(status)
        self.save()
        print('Prepared', self.manifest['scenario'], 'at', self.directory)
        print('Source:', commit, 'effective batch:', self.manifest['effective_batch'])
        print('Checkpoint selection:', CHECKPOINT_RULE)

    def source_state(self):
        def git(*arguments):
            return subprocess.check_output(['git', *arguments], cwd=self.repo, text=True)
        commit = git('rev-parse', 'HEAD').strip()
        diff = git('diff', '--binary', 'HEAD', '--')
        status = git('status', '--porcelain=v1', '--untracked-files=all')
        if commit != self.source_commit or len(commit) != 40 or diff or status:
            raise RuntimeError('Source must be clean at the full pinned source SHA')
        return commit, diff, status

    def save(self):
        (self.directory / 'run_manifest.json').write_text(json.dumps(self.manifest, indent=2) + '\n')

    def environment(self):
        env = os.environ.copy()
        entries = [str(self.hook_dir), str(self.repo)]
        entries += [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p and p not in entries]
        env.update(PYTHONPATH=os.pathsep.join(entries), P3_TENSORBOARD_NO_TF='1',
                   PYTHONUNBUFFERED='1', RUNNER_CAPTURING_LOG='1')
        if self.resume_checkpoint:
            # Only the explicitly selected trusted full-state training checkpoint.
            env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        return env

    def verify(self):
        self.source_state()
        if sha256(self.active_config) != self.manifest['config_sha256']:
            raise RuntimeError('Active config changed; prepare a new run and repeat all checks')
        if sha256(self.hook) != self.manifest['tensorboard_runtime_fix']['startup_sha256']:
            raise RuntimeError('Runtime hook changed; prepare a new run and repeat all checks')
        if versions() != self.manifest['versions']:
            raise RuntimeError('Package versions changed; prepare a new run and repeat all checks')
        if self.num_devices != self.manifest['num_devices']:
            raise RuntimeError('Device count changed; prepare a new run')
        if (self.resume_checkpoint or None) != self.manifest['resume_checkpoint']:
            raise RuntimeError('Resume selection changed; prepare a new run')

    def validate_resume(self, checkpoint, config, manifest):
        if any((checkpoint, config, manifest)) and not all((checkpoint, config, manifest)):
            raise ValueError('Resume needs checkpoint, active config and run manifest from the same run')
        if not checkpoint:
            return
        checkpoint, config, manifest = map(lambda p: Path(p).resolve(), (checkpoint, config, manifest))
        previous = json.loads(manifest.read_text())
        if previous['scenario'] != self.manifest['scenario'] or previous['source_commit'] != self.source_commit:
            raise ValueError('Resume scenario/source mismatch')
        if sha256(config) != previous['config_sha256'] or sha256(config) != self.manifest['config_sha256']:
            raise ValueError('Resume active config mismatch')
        if previous['tensorboard_runtime_fix']['startup_sha256'] != self.manifest['tensorboard_runtime_fix']['startup_sha256']:
            raise ValueError('Resume runtime-fix mismatch')
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        # Require the original run bundle, or an exact digest recorded at packaging.
        recorded = previous.get('checkpoint_files', {})
        relative = str(checkpoint.relative_to(manifest.parent)) if checkpoint.is_relative_to(manifest.parent) else None
        digest = sha256(checkpoint)
        if relative in recorded:
            if recorded[relative] != digest:
                raise ValueError('Resume checkpoint hash mismatch')
        elif digest not in recorded.values():
            if relative is None or not relative.startswith('training_output/checkpoints/'):
                raise ValueError('Keep checkpoint in its original run bundle beside config/manifest')
        self.manifest['resume_checkpoint_sha256'] = digest

    def execute(self, script, *arguments, log):
        run([sys.executable, '-u', script, *map(str, arguments)], cwd=self.repo,
            env=self.environment(), log_path=self.directory / log)

    def logger_smoke(self):
        self.manifest.update(logger_passed=False, preflight_passed=False, status='checking_logger')
        self.save()
        self.verify()
        output = self.directory / 'logger_smoke' / 'report.json'
        self.execute('tools/lsh_logger_smoke.py', '--output', output, log='logger_smoke.log')
        report = json.loads(output.read_text())
        if not report.get('passed'):
            raise RuntimeError('Logger smoke test did not pass')
        self.manifest.update(logger_passed=True, logger_smoke=report,
                             logger_report_sha256=sha256(output), status='logger_passed')
        self.save()

    def preflight(self):
        self.manifest.update(preflight_passed=False, status='checking_preflight')
        self.save()
        self.verify()
        if not self.manifest['logger_passed']:
            raise RuntimeError('Run the real TensorBoard logger smoke test first')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.num_devices:
            raise RuntimeError(f'Need {self.num_devices} visible GPU(s); GPU preflight has not passed')
        self.manifest['runtime_gpu'] = [dict(name=torch.cuda.get_device_name(i),
                total_memory=torch.cuda.get_device_properties(i).total_memory)
                for i in range(torch.cuda.device_count())]
        self.save()
        self.execute('test_lsh.py', '--output', self.directory / 'core_checks.json', log='core_checks.log')
        self.execute('tools/lsh_preflight.py', '--config', self.active_config,
                     '--output', self.directory / 'gpu_preflight.json', log='gpu_preflight.log')
        # Distributed protocol test uses small probes on every requested GPU.
        # If explicitly running one GPU, record the remaining GPU-DDP limitation.
        ddp_device = 'cuda' if self.num_devices > 1 else 'cpu'
        workers = self.num_devices if self.num_devices > 1 else 2
        run([sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
             f'--nproc-per-node={workers}', 'tools/lsh_ddp_smoke.py', '--config', str(self.active_config),
             '--output', str(self.directory / 'ddp_smoke.json'), '--device', ddp_device],
            cwd=self.repo, env=self.environment(), log_path=self.directory / 'ddp_smoke.log')
        checks = {name: json.loads((self.directory / f'{name}.json').read_text())
                  for name in ('core_checks', 'gpu_preflight', 'ddp_smoke')}
        for name in ('gpu_preflight', 'ddp_smoke'):
            if not checks[name].get('passed') or checks[name]['model'] != self.cfg['backbone']['name']:
                raise RuntimeError(f'Wrong/failed {name} for this model')
        self.verify()
        self.manifest.update(preflight_passed=True, status='preflight_passed', preflight=checks,
            data_shapes=checks['gpu_preflight']['dataset_samples'],
            preflight_report_hashes={name: sha256(self.directory / f'{name}.json') for name in checks},
            gpu_ddp_tested=ddp_device == 'cuda')
        self.save()
        print('Preflight passed for this model. Review memory and sample shapes before training.')

    def train(self):
        self.verify()
        if not self.manifest['logger_passed'] or not self.manifest['preflight_passed']:
            raise RuntimeError('Logger and per-model GPU/DDP preflight must pass before train')
        if sha256(self.directory / 'logger_smoke/report.json') != self.manifest['logger_report_sha256']:
            raise RuntimeError('Logger report changed; repeat checks')
        for name, expected in self.manifest['preflight_report_hashes'].items():
            if sha256(self.directory / f'{name}.json') != expected:
                raise RuntimeError('Preflight report changed; repeat checks')
        command = [sys.executable, '-u', 'multi_metric_train.py', '--config', str(self.active_config),
                   '--export_dir', str(self.directory / 'training_output'), '--devices', str(self.num_devices)]
        if self.resume_checkpoint:
            if sha256(self.resume_checkpoint) != self.manifest['resume_checkpoint_sha256']:
                raise RuntimeError('Resume checkpoint changed after preparation')
            command += ['--ckpt_path', self.resume_checkpoint, '--resume']
        self.manifest.update(training_command=command, status='training')
        self.save()
        try:
            run(command, cwd=self.repo, env=self.environment(), log_path=self.directory / 'training.log')
            self.manifest['status'] = 'complete'
        except BaseException as error:
            self.manifest.update(status='failed_or_interrupted', error=str(error))
            raise
        finally:
            self.package()

    def package(self):
        self.manifest['checkpoint_files'] = {str(p.relative_to(self.directory)): sha256(p)
            for p in (self.directory / 'training_output/checkpoints').glob('*.ckpt')}
        self.save()
        destination = self.directory.parent / f'{self.manifest["scenario"]}_{self.directory.name}_artifacts.zip'
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in self.directory.rglob('*'):
                relative = path.relative_to(self.directory)
                if path.is_file() and relative.parts[0] != 'source':
                    archive.write(path, str(relative))
        print('Artifacts:', destination)
        return destination
