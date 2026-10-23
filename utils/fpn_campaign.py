"""Locked Fuse campaign IDs and config loader (no training overrides)."""
from pathlib import Path
import yaml

SCENARIOS = {'control': 'FPN.Control', 'aex1': 'FPN.A-oA', 'aex2': 'FPN.A-1A', 'bex1': 'FPN.B-vani.P52', 'bex2': 'FPN.B-vani.P42', 'bex3': 'FPN.B-dpw.P52', 'bex4': 'FPN.B-dpw.P42', 'bex5': 'FPN.B-vani.P43', 'bex6': 'FPN.B-dpw.P43'}
ROOT = Path(__file__).resolve().parents[1]

def config(mode):
    return yaml.safe_load((ROOT/"configs/p3_fuse_fpn"/f"{SCENARIOS[mode]}.yaml").read_text())
