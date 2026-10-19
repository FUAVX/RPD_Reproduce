#!/usr/bin/env python3
"""Disposable AAA benchmarks; no campaign weights/optimizer/RNG are reused.

Inference: python scripts/benchmark_aaa.py --mode inference --output-dir RUN/benchmark
Training T4x2: python -m torch.distributed.run --standalone --nproc_per_node=2 \
 scripts/benchmark_aaa.py --mode training --output-dir RUN/benchmark
Default inference30/200 and training20/100 follow AAA v3. Explicit smaller
--size/--warmup/--iterations values are smoke tests, not campaign evidence.
"""
import argparse
from contextlib import contextmanager
import copy
import gc
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT/'configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml'
PLACEMENTS = [('control', None), ('early', 'P3_AAA_Ex1_Early'),
              ('late', 'P3_AAA_Ex2_Late'), ('all', 'P3_AAA_Ex3_All')]


@contextmanager
def isolated_rng(seed, device):
    py_state, np_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed);np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state);np.random.set_state(np_state)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def campaign_backend(cfg, mode):
    previous = torch.backends.cudnn.benchmark
    if mode == 'training':
        torch.backends.cudnn.benchmark = bool(cfg['train'].get('benchmark', True))
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


def measure(step, device, warmup, iterations, global_batch, pure_forward=False):
    import torch.distributed as dist
    for _ in range(warmup):
        step()
    synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iterations):
        synchronize(device)
        if pure_forward and device.type == 'cuda':
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record();step();end.record();synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            start = time.perf_counter();step();synchronize(device)
            times.append((time.perf_counter()-start)*1000)
    # Global DDP throughput uses the slowest rank per synchronized iteration.
    if dist.is_initialized():
        t = torch.tensor(times, device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        times = t.cpu().tolist()
    memory = {'allocated_MiB': torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else None,
              'reserved_MiB': torch.cuda.max_memory_reserved(device)/2**20 if device.type == 'cuda' else None}
    memories = [memory]
    if dist.is_initialized():
        memories = [None]*dist.get_world_size();dist.all_gather_object(memories, memory)
    mean = float(np.mean(times))
    return {'mean_ms': mean, 'P50_ms': float(np.percentile(times, 50)),
            'P95_ms': float(np.percentile(times, 95)), 'images_per_second': 1000*global_batch/mean,
            'warmup': warmup, 'measured': iterations, 'peak_memory_by_rank': memories,
            'timing': 'CUDA events' if pure_forward and device.type == 'cuda' else 'synchronized wall clock; max rank per step for DDP'}


def benchmark_candidate(cfg, mode, device, size, warmup, iterations, graph='deploy'):
    from models import get_backbone, get_criterion
    from models.rpdnet.RPD_Module import RPD_model_deploy
    from models.rpdnet.dpp import dpp_optimizer_groups
    from torch.profiler import profile, ProfilerActivity
    import torch.distributed as dist
    device = torch.device(device)
    world = dist.get_world_size() if dist.is_initialized() else 1
    batch = cfg['train']['batch_size'] if mode == 'training' else 1
    with isolated_rng(cfg['seed'], device), campaign_backend(cfg, mode):
        model = get_backbone(cfg)
        params = sum(p.numel() for p in model.parameters())
        if mode == 'inference' and graph == 'deploy':
            model = RPD_model_deploy(model.eval(), do_copy=True)
        model = model.to(device).float()
        generator = torch.Generator(device=device).manual_seed(cfg['seed'])
        x = torch.randn(batch, 3, size, size, generator=generator, device=device)
        if mode == 'inference':
            model.eval()
            with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU], with_flops=True) as p:
                model(x)
            flops = sum(e.flops for e in p.key_averages() if e.key == 'aten::conv2d')
            del p
            if not flops:
                raise RuntimeError('Functional-convolution profiling failed.')
            def step():
                with torch.inference_mode():
                    model(x)
        else:
            model.train()
            if dist.is_initialized():
                model = torch.nn.parallel.DistributedDataParallel(model,
                    device_ids=[device.index] if device.type == 'cuda' else None,
                    find_unused_parameters=False)
            criterion = get_criterion(cfg).to(device)
            optimizer = torch.optim.Adam(dpp_optimizer_groups(model, cfg['train']['learning_rate'], cfg['train']['weight_decay']))
            target = torch.randint(0, 3, (batch, size, size), generator=generator, device=device)
            def step():
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), target, mode='train', mask_keep=target != 255)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite benchmark WDice.')
                loss.backward();optimizer.step()
        measured = measure(step, device, warmup, iterations, batch*world, pure_forward=mode == 'inference')
        result = {**measured, 'resolution': [size, size], 'batch_per_rank': batch,
                  'world_size': world, 'trainable_params': params,
                  'cudnn_benchmark': torch.backends.cudnn.benchmark,
                  'graph': graph if mode == 'inference' else 'unfused_train',
                  'scope': 'synthetic compute only; no image IO/data loader; controlled untrained disposable weights',
                  'status': 'GPU_measured' if device.type == 'cuda' else 'CPU_smoke_not_GPU_evidence'}
        if mode == 'inference':
            result.update(Conv_MACs=flops//2, Conv_FLOPs=flops,
                          MAC_scope='Executed aten::conv2d including functional RPD and AAA; excludes max comparisons, padding, normalization, interpolation')
    # Local objects, optimizer state and tensor references die on return.
    return result


def main():
    import os
    import torch.distributed as dist
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['inference', 'training'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--graph', choices=['deploy', 'unfused'], default='deploy')
    parser.add_argument('--size', type=int)
    parser.add_argument('--warmup', type=int)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--placements', nargs='+', choices=[x[0] for x in PLACEMENTS], default=[x[0] for x in PLACEMENTS])
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if 'control' not in args.placements:
        parser.error('Include control for paired benchmark comparisons.')
    warmup = args.warmup if args.warmup is not None else (30 if args.mode == 'inference' else 20)
    iterations = args.iterations if args.iterations is not None else (200 if args.mode == 'inference' else 100)
    sizes = [args.size] if args.size else ([768, 1024] if args.mode == 'inference' else [768])
    if min(sizes) < 32 or warmup < 1 or iterations < 1:
        parser.error('size>=32 and positive warmup/iterations required.')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested; CPU smoke requires explicit --device cpu.')
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', device.index or 0)))
        torch.cuda.set_device(device)
    if world > 1:
        if args.mode != 'training':
            parser.error('Inference benchmark is single-GPU.')
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    output = Path(args.output_dir);output.mkdir(parents=True, exist_ok=True)
    report = {'mode': args.mode, 'precision': 'FP32', 'torch': torch.__version__,
              'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'settings': vars(args), 'world_size': world, 'results': {}, 'status': 'running'}
    report['deployment_certified'] = False
    report['quality_evaluation'] = 'not measured; disposable synthetic compute benchmark'
    report['campaign_protocol_match'] = (device.type == 'cuda' and
        ((args.mode == 'inference' and warmup == 30 and iterations == 200 and sizes == [768,1024] and world == 1)
         or (args.mode == 'training' and warmup == 20 and iterations == 100 and sizes == [768] and world == 2 and 'T4' in report['gpu'])))
    base = yaml.safe_load(BASE_CONFIG.read_text())
    try:
        for name, scenario in PLACEMENTS:
            if name not in args.placements:
                continue
            cfg = yaml.safe_load((ROOT/f'configs/p3_scenarios/{scenario}.yaml').read_text()) if scenario else copy.deepcopy(base)
            control = copy.deepcopy(cfg);control['experiment'] = base['experiment'];control['backbone'].pop('downsampling', None)
            if control != base:
                raise ValueError('AAA benchmark recipe drift from P2.')
            for size in sizes:
                key = f'{name}_{size}'
                print('[BENCH]', args.mode, key, flush=True)
                report['results'][key] = benchmark_candidate(cfg, args.mode, device, size, warmup, iterations, args.graph)
                reference = report['results'][f'control_{size}']['mean_ms']
                report['results'][key]['latency_delta_percent_vs_control'] = 100*(report['results'][key]['mean_ms']/reference-1)
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if not dist.is_initialized() or dist.get_rank() == 0:
            (output/f'benchmark_{args.mode}.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
