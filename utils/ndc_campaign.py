"""NOW is Control + one NDC-Full experiment; no FUT variants."""
from pathlib import Path
import yaml
SCENARIOS = {"control":"NDC-Control", "ex1":"NDC-Ex1"}
ROOT = Path(__file__).resolve().parents[1]

def config(mode):
    return yaml.safe_load((ROOT/"configs/p3_fuse_ndc"/f"{SCENARIOS[mode]}.yaml").read_text())
