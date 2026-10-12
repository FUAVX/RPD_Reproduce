# Fresh EncoderHalf / DecoderHalf training

Use `configs/p2_scenarios/P2_EncoderHalf.yaml` or `P2_DecoderHalf.yaml` with the shared `kaggle_train_rpd.ipynb` on branch `P2_ModiRPD_EnDe` of `FUAVX/WeedSegment_DPL302m-FALL26`. The notebook clones branch `p2_ende`, checks the dataset and models in a separate process, then starts exactly two ranks with torchrun. Clone/preflight failures stop the cell.

## Handoff section 3: architecture

`models/rpdnet/RepDWNetEnDe.py` reuses the B1 Run 5 `DoubleConv`, `Down_repdw`, `DoubleConv_down_repdw`, `OutConv`, `RepDWNet.forward` and `Up_repdw.forward`. Only channel widths change. The original DW 1x1 branch and conditional identity remain unchanged; no skip projections are added. The pointwise RPD in the pinned B1 implementation has four branches; it is not replaced with the original RPDNet backbone or teammate DW/PW ablation code.

| Model | Stem | Encoder output | Decoder concat | Decoder mid | Decoder output | Head |
|---|---|---|---|---|---|---|
| EncoderHalf | 16 | 16,32,64,64 | 128,96,48,32 | 128,64,32,16 | 64,32,16,16 | 16→3 |
| DecoderHalf | 16 | 32,64,128,128 | 256,96,48,24 | 64,32,16,8 | 32,16,8,8 | 8→3 |

The factory rejects channel tables or block settings that differ from these experiments. Baseline block files are compared byte-for-byte with source revision `b53d8ba1ddfd41ebd37c9021b1cb6cdee332084e`; their hashes are recorded in the manifest. The actual runner commit is recorded separately.

## Handoff section 4: fresh recipe

The runner validates resolved config after CLI overrides: seed 1682409321, Adam, LR/WD 2e-4, cosine with the existing B1 five-epoch warmup, 100 epochs, no early stopping, FP32, two GPUs/DDP, batch 4 per GPU, accumulation 1 and effective batch 8. Normalization/augmentations are compared to the B1 Run 5 YAML; train crop is 768², val crop 1024², validation interval five epochs.

`--resume`, `--ckpt_path`, pretrained initialization and reused training-artifact directories are rejected. The notebook creates a new run ID every launch. Interruptions require another fresh run. Last checkpoints are still saved for artifact preservation.

Train and validation retain `drop_last=True`. For 772 validation images, two ranks and batch 4, 768 images are evaluated. This is not a full 772-image evaluation. Dataset preflight checks that every train/val image has a semantic mask; loader counts/settings are checked at fit start. I/O retry is disabled for both models in these configs; a read error fails the run, without skipping or replacing a sample.

## Handoff sections 5–6: checks and artifacts

`tools/ende_preflight.py` checks the factory class, each stage's channels, decoder mid/concat, baseline stem and DecoderHalf encoder structure, even/odd input shapes, finite output, backward gradients and strict checkpoint loading followed by B1 fusion. Its process does not change training RNG or BN state. Fusion is checked in FP32 against the output's dynamic range (scaled max error <1e-5); this is not a guarantee about metrics on a trained checkpoint.

`FreshEnDeCallback` verifies WORLD_SIZE=2, LOCAL_RANK, two CUDA GPUs, FP32, new Adam state, epoch/global_step zero and loader protocol. The manifest records scenario/run, baseline and runtime revisions, notebook commit and executed-cell hash, channel table, parameter count, recipe, dataset counts, validation coverage and checkpoint paths/epochs/scores.

Sanity metric compute/log/reset still executes on all ranks. History CSV and classwise YAML export skip sanity and write only on rank 0. Lightning's synchronized `log_dir` is resolved before the rank guard. Config YAML/summary/curves also write on rank 0; process logs have separate rank filenames. The final manifest is complete only with train epoch 99 and validation epochs 4,9,…,99. Exceptions/incomplete fits are marked failed. Best mIoU, best Weed IoU and last checkpoints are retained along with existing baseline checkpoint callbacks.

Run output:

```text
log_dir/<scenario>/<baseline_revision>/<new_run_id>/
  preflight.json
  experiment_manifest.json
  logs/
  lightning_logs/
  checkpoints/
  val_history.csv
  val_curves.png
  val_summary.txt
```

## Verification performed

- Both real factory models passed stage hooks for 64×64 and 65×67 RGB inputs, finite outputs, backward gradients, strict loading and fusion checks.
- Parameter counts: EncoderHalf 98,611; DecoderHalf 149,435.
- Both passed a two-rank CPU DDP smoke run of ten epochs on synthetic data, using the real loss, segmentation module and checkpoint callbacks. CSV contains exactly `[4,9]`, without sanity rows; both best metric checkpoints and last checkpoint were saved. Rank 1 has no exported history.
- Unit tests passed for recipe/resume guards, missing-mask failure, directory preservation and complete/failed manifest lifecycle. Manifest GPU checks in unit tests use a mock; actual two-GPU runtime is checked on Kaggle.
- The shared notebook passed nbformat validation and IPython-transformed Python parsing. The training CLI imports successfully.

Tests used Python 3.11, PyTorch 2.12.1 CPU, Lightning 1.9.5 and TorchMetrics 0.11.4 in an isolated environment. No PhenoBench training or 100-epoch GPU run has been performed here. The CPU smoke results are not experiment metrics.

Reproduce checks:

```bash
python tools/ende_preflight.py --config configs/p2_scenarios/P2_EncoderHalf.yaml
python tools/ende_preflight.py --config configs/p2_scenarios/P2_DecoderHalf.yaml
python tests/test_ende_contract.py
torchrun --standalone --nproc_per_node=2 tests/ende_smoke.py --config configs/p2_scenarios/P2_EncoderHalf.yaml --output /tmp/ende-smoke-new-en
torchrun --standalone --nproc_per_node=2 tests/ende_smoke.py --config configs/p2_scenarios/P2_DecoderHalf.yaml --output /tmp/ende-smoke-new-de
```

## Startup and live logging (2026-10-13)

The shared notebook now creates `/kaggle/working/notebook_logs/<startup_id>/launcher.log` and `runner_status.json` before installing or importing third-party dependencies. NVIDIA diagnostics have a 30-second deadline and dependencies a 1,200-second deadline. A failure stops execution and records its stage.

For the selected scenario, logging starts before clone in `<run_dir>/logs/`. Git/config-import/GPU-probe steps have 180-second deadlines; preflight has a 900-second deadline. Each process combines stdout/stderr into a live stage log and streams it into notebook output. It handles output without a newline. Timeouts terminate the complete child process group. `runner_status.json` records the current stage, PID, elapsed time, output bytes and failure details.

En/De has a 900-second startup deadline until rank 0 emits the actual `training started` callback marker (after imports, DDP initialization and sanity validation). Once that marker arrives, training has no total-duration deadline. `logs/training.log` is updated continuously with both torchrun ranks' output; separate `train_*_rank0.log` and `rank1.log` remain available. The launcher emits a heartbeat every 30 seconds showing how long it has waited and when the last process output arrived. A heartbeat only proves that the launcher is alive; it does not prove epochs are advancing. Imports, preflight subchecks, fit setup, sanity checks and epoch boundaries emit explicit markers. If startup stalls, a traceback is dumped after 300 seconds and repeated until training starts.

The direct CLI also updates `training.log` live from rank 0. It no longer copies a one-time snapshot at startup. No channel table, baseline block, augmentation, loss or training recipe is changed by this logging update. The cause of the user's eight-hour silent Kaggle run has not been reproduced; these diagnostics identify the stage on the next execution.

Logging verification: four runner unit tests passed (live stdout/stderr without a newline, heartbeat, subprocess errors, timeout cleanup and startup-marker deadline release); TeeStream live-write regression and fresh-contract tests passed. Both real En/De preflights and both two-rank CPU DDP smoke runs passed through the streaming runner. Actual training CLI failure paths for both scenarios persisted boot markers, rank logs and full recipe-guard tracebacks. Notebook format/syntax and embedded helper were validated. Two-GPU Kaggle training was not executed in these checks.
