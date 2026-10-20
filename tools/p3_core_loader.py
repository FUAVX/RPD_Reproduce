"""Dependency-light test isolation for actual model/loss files, never for training.

The source repository eagerly imports every backbone in models/__init__.py.
This helper bypasses package initializers to test the torch-only core when
Lightning/timm are unavailable. Runtime integration is a separate preflight.
"""
import importlib
import sys
import types
from pathlib import Path


def load_core():
    root = Path(__file__).resolve().parents[1]
    if "models" in sys.modules:
        raise RuntimeError("Core isolation must run in a fresh process before importing models.")
    for name, folder in (("models", root / "models"),
                         ("models.rpdnet", root / "models" / "rpdnet")):
        package = types.ModuleType(name)
        package.__path__ = [str(folder)]
        sys.modules[name] = package
    baseline = importlib.import_module("models.rpdnet.RepDWNet").RepDWNet
    h8 = importlib.import_module("models.rpdnet.RepDWNetH8").RepDWNetH8
    fusion = importlib.import_module("models.rpdnet.RepDWNetH4H8Fusion").RepDWNetH4H8Fusion
    losses = importlib.import_module("models.losses")
    return baseline, h8, fusion, losses
