# Phase 3 integration verification — 2026-10-20 (Asia/Saigon)

Repository: `https://github.com/FUAVX/RPD_Reproduce.git`.
Original checkout: `/home/trandouyenphuong/RPD_Reproduce`, branch `p2_ende`,
HEAD `1f39156`, no tracked/untracked changes before integration; ignored caches
preserved. No applicable AGENTS.md was found in project/ancestor paths.

Independent worktree: `/home/trandouyenphuong/RPD_Phase3_PhuongTU`,
branch `codex/p3-srr`, based directly on `origin/f/wdice-loss` at
`d77d518b8f650443216070fa76742db76e41b8ac` (confirmed against GitHub after fetch).
Runtime source commit: `9c9446c360ddaaff78db208c9cc3cb9b810a9b77`.
Notebook/guide are a subsequent commit whose hash can be read with `git log`.

## Scope review

Two model implementations come from the handoff, with only EOF formatting
normalization. Factory adds two imports/constructors. No modifications to B1,
RPD/RepDW blocks, losses, datasets, training loop or P2 recipe. These existing
files were also compared byte-for-byte against the WDice base. Both P3 YAMLs
match all baseline top-level recipe fields except experiment/backbone.
No main merge, force push, source ZIP overlay or long training.

The source tests gained an external output option so normal notebook checks do
not dirty the pinned checkout; exact recipe equality and simultaneous exchange
checks were added. GPU preflight also checks unused/nonfinite gradients and
seeds Python/NumPy/torch via Lightning. The two models were not reimplemented.

## Executed

Isolated environment: `/home/trandouyenphuong/.venvs/rpd-p3`, Python 3.11.16,
PyTorch 2.7.1+cpu, torchvision 0.22.1+cpu, Lightning 1.9.5,
torchmetrics 0.11.4, timm 0.9.16, NumPy 1.26.4, setuptools 69.5.1.
Dependencies were installed with the notebook's dependency list, preserving
CPU torch/torchvision. `pip check` passed. Full package versions are in
`environment.txt`.

- `python test_p3.py --output docs/p3/normal_import_checks.json`: passed.
- `python test_p3.py --core-only --output docs/p3/core_checks.json`: passed.
- `python test_wdice.py`: all four original tests passed.
- `python multi_metric_train.py --help`: normal imports and CLI passed.
- `python tools/p3_preflight.py --help`: normal imports and CLI passed;
  this does not execute GPU checks.
- `python tools/p3_notebook_checks.py`: both notebooks passed. Uses a local
  Git clone at the real source commit and an empty dataset directory layout.
  Executes preparation cell, checks recipe/provenance/sparse checkout, rejects
  dirty source and incorrect resume, checks the preflight gate. Fresh/resume
  training CLI and artifacts are checked with a mocked subprocess; no training
  or actual dataset loading is performed by this notebook test.
- Both notebooks passed nbformat validation; every code cell compiled, with
  cleared outputs and execution counts.
- Git whitespace check passed with `cr-at-eol` enabled for the existing CRLF
  factory; new files have normal final newlines.

All four tested architectures produce `[2,3,64,64]` and `[1,3,65,79]`; all
trainable parameters have finite gradients. Params: B1/PA1 189827,
PA2 333891, C0 292611. Stage shapes and deploy diagnostics are in the JSONs.
Deploy diagnostics use random uncalibrated logits and normalized error;
absolute max error is reported and does not certify trained-checkpoint deploy.

## Remaining runtime checks

GPU, CUDA/DDP, actual Kaggle runtime/dependency state, full dataset, native
validation memory, checkpoint resume and segmentation quality are untested.
Notebook GPU/full-batch/data preflight must pass before training. It measures
one GPU with the actual per-GPU batch and the first validation sample shape;
it does not establish DDP or whole-dataset memory safety. No training result,
small-region coverage, miss rate, FLOPs or GPU latency is claimed here.
