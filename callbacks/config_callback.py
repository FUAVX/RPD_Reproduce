import os
from typing import Dict

import yaml
from pytorch_lightning.callbacks import Callback


class ConfigCallback(Callback):
    """ Callback to save the config file of a model. """

    def __init__(self, cfg: Dict):
        self.cfg = cfg

    def setup(self, trainer, pl_module, stage=None) -> None:
        # trainer.log_dir can broadcast its value: every rank must resolve it
        # before the rank-zero guard to keep DDP collective order consistent.
        export_dir = os.path.join(trainer.log_dir, 'configuration')
        if not trainer.is_global_zero:
            return
        os.makedirs(export_dir, exist_ok=True)

        fpath = os.path.join(export_dir, 'config.yaml')
        with open(fpath, 'w') as ostream:
            yaml.dump(self.cfg, ostream)