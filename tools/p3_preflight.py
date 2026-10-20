"""Normal-import, actual-config synthetic GPU check before long Kaggle training."""
import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from models import get_backbone, get_criterion
from datasets import get_data_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU preflight requires a GPU runtime; select a Kaggle GPU accelerator.")
    from pytorch_lightning import seed_everything
    seed_everything(cfg["seed"], workers=True)
    model = get_backbone(cfg).to(device)
    criterion = get_criterion(cfg).to(device)
    report = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  params=sum(p.numel() for p in model.parameters()),
                  note="Synthetic local per-device batch; not a dataset/DDP training benchmark.")
    # Exercise the actual dataset/transforms and label remapping as well.
    data_module = get_data_module(cfg)
    data_module.setup("fit")
    report["dataset_samples"] = {}
    for split, dataset in (("train", data_module.train_ds), ("val", data_module.val_ds)):
        sample = dataset[0]
        image, labels = sample["input_image"], sample["anno"]
        assert image.ndim == 3 and image.shape[0] == 3
        assert labels.shape == image.shape[-2:]
        valid = labels != 255 if split == "train" else torch.ones_like(labels, dtype=torch.bool)
        if not torch.any(valid):
            raise RuntimeError(f"First {split} sample has no valid labeled pixels.")
        if torch.any(valid & ((labels < 0) | (labels >= cfg["backbone"]["num_classes"]))):
            raise RuntimeError(f"Unexpected {split} labels after preprocessing.")
        report["dataset_samples"][split] = dict(image_shape=list(image.shape),
                                                label_ids=labels.unique().tolist())
    del data_module, sample, image, labels, valid
    crop = cfg["train"]["geometric_data_augmentations"]["random_crop"]
    h, w, batch = crop["height"], crop["width"], cfg["train"]["batch_size"]
    torch.cuda.reset_peak_memory_stats()
    model.train()
    inputs = torch.randn(batch, 3, h, w, device=device)
    target = torch.randint(0, cfg["backbone"]["num_classes"], (batch, h, w), device=device)
    logits = model(inputs)
    assert logits.shape == (batch, cfg["backbone"]["num_classes"], h, w)
    loss = criterion(logits, target, mode="train", mask_keep=target != 255)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic training loss.")
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    invalid = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if unused or invalid:
        raise RuntimeError(f"Invalid gradients: unused={unused}, nonfinite={invalid}")
    report["all_trainable_parameters_used"] = True
    # Materialize AdamW state too; forward/backward alone underestimates memory.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    optimizer.step()
    torch.cuda.synchronize()
    report["train_shape"] = list(logits.shape)
    report["train_loss"] = float(loss.detach())
    report["train_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    optimizer.zero_grad(set_to_none=True)
    del optimizer, inputs, target, logits, loss
    gc.collect()
    torch.cuda.empty_cache()
    # The supplied PDC.get_val_item returns native validation image dimensions;
    # it does not apply the configured geometric center_crop. Preserve baseline
    # behavior and measure the shape actually emitted by the dataset.
    h, w = report["dataset_samples"]["val"]["image_shape"][-2:]
    batch = cfg["val"]["batch_size"]
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        logits = model(torch.randn(batch, 3, h, w, device=device))
    assert logits.shape == (batch, cfg["backbone"]["num_classes"], h, w)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite synthetic validation logits.")
    torch.cuda.synchronize()
    report["validation_shape"] = list(logits.shape)
    report["validation_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print("P3 GPU preflight passed.", flush=True)


if __name__ == "__main__":
    main()
