"""Locked ATG campaign IDs and config loader; no training overrides."""
from pathlib import Path
import yaml
SCENARIOS = {'control': 'ATG-Control', 'ex1': 'ATG-Ex1', 'ex2': 'ATG-Ex2', 'ex3': 'ATG-Ex3', 'ex4': 'ATG-Ex4', 'ex5': 'ATG-Ex5', 'ex6': 'ATG-Ex6', 'ex7': 'ATG-Ex7', 'ex8': 'ATG-Ex8'}
ROOT = Path(__file__).resolve().parents[1]

def config(mode):
    return yaml.safe_load((ROOT/"configs/p3_fuse_atg"/f"{SCENARIOS[mode]}.yaml").read_text())
