#!/usr/bin/env python3
"""Synthetic FP32 Control vs selected LIP microbenchmark; one GPU per process.

This measures model compute, not data loading or two-GPU DDP throughput.
python scripts/benchmark_lip.py --device cuda --output-dir RUN/benchmark
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import yaml


def convolution_macs(model, x):
    counts = []
    handles = []
    def hook(layer, inputs, output):
        counts.append(output.numel() * (layer.in_channels // layer.groups)
                      * layer.kernel_size[0] * layer.kernel_size[1])
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                handles.append(module.register_forward_hook(hook))
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return sum(counts)


def measure(step, device, batch_size, warmup, iterations):
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        step()
    sync()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        step()
        sync()
        times.append(time.perf_counter() - start)
    seconds = statistics.mean(times)
    return {'mean_ms': seconds * 1000, 'min_ms': min(times)*1000, 'max_ms': max(times)*1000,
            'images_per_second': batch_size / seconds,
            'peak_allocated_MiB': (torch.cuda.max_memory_allocated(device) / 2**20
                                   if device.type == 'cuda' else None),
            'peak_reserved_MiB': (torch.cuda.max_memory_reserved(device) / 2**20
                                  if device.type == 'cuda' else None)}


def benchmark(cfg, args):
    from models import get_backbone, get_criterion
    from models.rpdnet.dpp import dpp_optimizer_groups
    torch.manual_seed(cfg['seed'])
    device = torch.device(args.device)
    model = get_backbone(cfg).to(device).float().eval()
    x = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    target = torch.randint(0, cfg['backbone']['num_classes'],
                           (args.batch_size, args.size, args.size), device=device)
    result = {'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    result['conv_MACs_per_image'] = convolution_macs(model, x[:1])
    result['conv_FLOPs_per_image'] = 2 * result['conv_MACs_per_image']
    def infer():
        with torch.inference_mode():
            model(x)
    result['inference'] = measure(infer, device, args.batch_size, args.warmup, args.iterations)
    model.train()
    criterion = get_criterion(cfg).to(device)
    optimizer = torch.optim.Adam(dpp_optimizer_groups(model, cfg['train']['learning_rate'],
                                                     cfg['train']['weight_decay']))
    def train():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), target, mode='train', mask_keep=target != 255)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite WDice loss during microbenchmark.')
        loss.backward()
        optimizer.step()
    result['training_iteration'] = measure(train, device, args.batch_size, args.warmup, args.iterations)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-config', default='configs/p2_scenarios/P2_Ex7_RepDWNet_WDice-Loss.yaml')
    parser.add_argument('--lip-config', default='configs/p3_scenarios/P3_LIP_Ex1_Early.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if min(args.size, args.batch_size, args.warmup, args.iterations) < 1 or args.size < 32:
        parser.error('size>=32 and positive batch/warmup/iterations required')
    control = yaml.safe_load(Path(args.control_config).read_text())
    lip = yaml.safe_load(Path(args.lip_config).read_text())
    inherited = copy.deepcopy(lip)
    ds = inherited['backbone'].pop('downsampling')
    inherited['experiment'] = control['experiment']
    placements = {(1, 2): 'early', (3, 4): 'late', (2, 3): 'middle',
                  (1, 2, 3): 'first3', (1, 2, 3, 4): 'all'}
    placement = placements.get(tuple(ds.get('stages', [])))
    if (inherited != control or placement is None
            or ds != {'variant': 'lip_bottleneck', 'stages': list(ds.get('stages', []))}):
        raise ValueError('Benchmark requires the unchanged P2 recipe and a supported LIP placement.')
    if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; choose --device cpu explicitly for a smoke test.')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': vars(args), 'torch': torch.__version__, 'precision': 'FP32',
              'device_name': torch.cuda.get_device_name(torch.device(args.device))
                             if torch.device(args.device).type == 'cuda' else 'CPU',
              'scope': 'synthetic single-device pre-deploy graph, no dataloader/DDP overhead',
              'mac_scope': 'Conv2d only; excludes IN, gating, aggregation, comparisons and backward',
              'placement': placement, 'stages': ds['stages'], 'results': {}, 'status': 'running'}
    try:
        for name, cfg in [('control', control), ('lip_'+placement, lip)]:
            report['results'][name] = benchmark(cfg, args)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Control vs selected LIP microbenchmark', '',
             f"Device: {report['device_name']}; FP32; batch={args.batch_size}; size={args.size}.", '',
             'Synthetic single-device measurement; excludes data loading and DDP.', '',
             '| Model | Params | Conv MAC/image | Inference ms | Train iteration ms | Train peak MiB |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in report['results'].items():
        lines.append(f"| {name} | {row['trainable_params']} | {row['conv_MACs_per_image']} | "
                     f"{row['inference']['mean_ms']:.3f} | {row['training_iteration']['mean_ms']:.3f} | "
                     f"{row['training_iteration']['peak_allocated_MiB']} |")
    (output / 'benchmark.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
