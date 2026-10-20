"""PhuongTU — two real DDP optimizer steps; synthetic tensors, not a run."""
import argparse
import hashlib
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml
from models import get_backbone, get_criterion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rank, local_rank, world_size = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world_size < 2:
        raise RuntimeError('Use torch.distributed.run with at least two workers')
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda', local_rank) if args.device == 'cuda' else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=3))
    try:
        model = get_backbone(cfg).to(device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank] if device.type == 'cuda' else None,
                                      find_unused_parameters=False)
        criterion = get_criterion(cfg).to(device)
        optimizer = torch.optim.Adam(ddp.parameters(), lr=cfg['train']['learning_rate'],
                                     weight_decay=cfg['train']['weight_decay'])
        torch.manual_seed(cfg["seed"] + rank + 1)  # Different inputs per rank.
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            target = torch.randint(0, 3, (2, 64, 64), device=device)
            target[:, 0, 0] = 255
            logits = ddp(torch.randn(2, 3, 64, 64, device=device))
            loss = criterion(logits, target, mode='train', mask_keep=target != 255)
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in model.parameters() if p.requires_grad)
            optimizer.step()
            losses.append(float(loss.detach()))
        from tensorboard.compat import tf
        _ = tf.io.gfile  # Resolve TensorBoard's LazyLoader before checking its backend.
        assert os.environ.get('P3_TENSORBOARD_NO_TF') == '1'
        assert 'tensorboard.compat.notf' in sys.modules
        assert tf.__name__ == 'tensorboard.compat.tensorflow_stub'
        assert 'tensorflow' not in sys.modules and 'jax' not in sys.modules
        parameter_digest = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes()
                                                   for p in model.parameters())).hexdigest()
        report = dict(rank=rank, steps=2, loss=losses, parameter_sha256=parameter_digest, all_parameters_have_finite_gradients=True,
                      tensorboard_backend=tf.__name__, startup_hook_inherited=True)
        reports = [None] * world_size
        dist.all_gather_object(reports, report)
        assert len({r["parameter_sha256"] for r in reports}) == 1, "DDP parameters diverged"
        if rank == 0:
            result = dict(author='PhuongTU', passed=True, model=cfg['backbone']['name'],
                          device=args.device, backend=dist.get_backend(), world_size=world_size,
                          find_unused_parameters=False, synthetic_shape=[2, 3, 64, 64], ranks=reports,
                          gpu_full_batch_or_dataset_test=False)
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
