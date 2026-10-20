"""PhuongTU — isolated real-data full-batch GPU preflight for one LSH model."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from pytorch_lightning import seed_everything
import yaml
from datasets import get_data_module
from models import get_backbone, get_criterion


def sample_report(sample, split, num_classes):
    image, labels = sample['input_image'], sample['anno']
    assert image.ndim == 3 and image.shape[0] == 3
    assert labels.shape == image.shape[-2:]
    valid = labels != 255 if split == 'train' else torch.ones_like(labels, dtype=torch.bool)
    if not valid.any() or (valid & ((labels < 0) | (labels >= num_classes))).any():
        raise RuntimeError(f'Unexpected {split} labels; preserve the baseline mask semantics')
    return dict(image_shape=list(image.shape), label_ids=labels.unique().tolist(),
                filename=sample['fname'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This preflight requires an actual GPU and PhenoBench input')
    cfg = yaml.safe_load(Path(args.config).read_text())
    assert cfg['backbone']['name'] in ('RepDWNet', 'RepDWNetLSHL1', 'RepDWNetLSHL2')
    assert cfg['train']['precision'] == 32
    seed_everything(cfg['seed'], workers=True)
    device = torch.device('cuda:0')
    data = get_data_module(cfg)
    data.setup('fit')
    train_sample, val_sample = data.train_ds[0], data.val_ds[0]
    report = dict(author='PhuongTU', model=cfg['backbone']['name'], torch=torch.__version__,
                  device=torch.cuda.get_device_name(0), passed=False,
                  data_path=cfg['data']['path_to_dataset'],
                  dataset_samples={split: sample_report(sample, split, cfg['backbone']['num_classes'])
                                   for split, sample in [('train', train_sample), ('val', val_sample)]},
                  scope='First real train/val samples repeated to per-GPU batch; not full dataset or DDP')
    crop = cfg['train']['geometric_data_augmentations']['random_crop']
    assert train_sample['input_image'].shape[-2:] == (crop['height'], crop['width'])
    model, criterion = get_backbone(cfg).to(device), get_criterion(cfg).to(device)
    # Match the optimizer class/options in SegmentationNetwork.configure_optimizers.
    # This one-step memory probe does not replace the original training scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'],
                                 weight_decay=cfg['train']['weight_decay'])
    report['optimizer'] = type(optimizer).__name__
    batch = cfg['train']['batch_size']
    images = train_sample['input_image'].unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
    labels = train_sample['anno'].unsqueeze(0).repeat(batch, 1, 1).to(device)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    logits = model(images)
    assert logits.shape == (batch, 3, *images.shape[-2:])
    loss = criterion(logits, labels, mode='train', mask_keep=labels != 255)
    if not torch.isfinite(loss):
        raise RuntimeError('Non-finite real-sample training loss')
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    invalid = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if unused or invalid:
        raise RuntimeError(f'Invalid gradients: unused={unused}, nonfinite={invalid}')
    optimizer.step()
    torch.cuda.synchronize()
    report.update(train_shape=list(logits.shape), train_loss=float(loss.detach()),
                  all_parameters_have_finite_gradients=True,
                  train_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    optimizer.zero_grad(set_to_none=True)
    del optimizer, images, labels, logits, loss, train_sample
    gc.collect()
    torch.cuda.empty_cache()
    batch = cfg['val']['batch_size']
    images = val_sample['input_image'].unsqueeze(0).repeat(batch, 1, 1, 1).to(device)
    labels = val_sample['anno'].unsqueeze(0).repeat(batch, 1, 1).to(device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(images)
        assert logits.shape == (batch, 3, *images.shape[-2:])
        # Match validation caller: no explicit mask. Ignore labels must fail here.
        val_loss = criterion(logits, labels, mode='val')
    assert torch.isfinite(logits).all() and torch.isfinite(val_loss)
    torch.cuda.synchronize()
    report.update(validation_shape=list(logits.shape), validation_loss=float(val_loss),
                  validation_peak_allocated_bytes=torch.cuda.max_memory_allocated(), passed=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
