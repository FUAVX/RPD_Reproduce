"""Locked CFF campaign: five experiments + optional P2 Control."""
from pathlib import Path
import yaml
SCENARIOS = {'control': 'CFF-Control', 'ex1': 'CFF-Ex1', 'ex2': 'CFF-Ex2', 'ex3': 'CFF-Ex3', 'ex4': 'CFF-Ex4', 'ex5': 'CFF-Ex5'}
DISPLAY_NAMES = {'ex1': 'CFF-FullD2', 'ex2': 'CFF-Dila1', 'ex3': 'CFF-Dila3', 'ex4': 'CFF-BNonly', 'ex5': 'CFF-IdenSkip', 'control': 'P2_WDice'}
ROOT = Path(__file__).resolve().parents[1]

def config(mode):
    return yaml.safe_load((ROOT/'configs/p3_fuse_cff'/f'{SCENARIOS[mode]}.yaml').read_text())
