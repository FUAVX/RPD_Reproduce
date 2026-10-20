"""PhuongTU — real TensorBoard event IO and inherited startup-hook check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ.get('P3_TENSORBOARD_NO_TF') == '1'
    assert 'tensorboard.compat.notf' in sys.modules, 'Startup hook was not loaded'
    import numpy
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from tensorboard.compat import tf
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from pytorch_lightning.loggers import TensorBoardLogger
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    backend = tf.__name__
    assert backend == 'tensorboard.compat.tensorflow_stub', backend
    event_dir = output.parent / 'events'
    writer = SummaryWriter(str(event_dir))
    writer.add_scalar('probe/loss', 1., 0)
    writer.add_scalars('probe/group', {'train': 1., 'val': .5}, 0)
    writer.flush()
    writer.close()
    events = EventAccumulator(str(event_dir)).Reload()
    assert events.Scalars('probe/loss')[0].value == 1.
    grouped = {}
    for child in event_dir.iterdir():
        if child.is_dir():
            grouped[child.name] = EventAccumulator(str(child)).Reload().Scalars('probe/group')[0].value
    assert grouped == {'probe_group_train': 1., 'probe_group_val': .5}, grouped
    logger = TensorBoardLogger(str(output.parent), name='lightning_probe', version=0)
    logger.log_hyperparams({'scenario': 'LSH_logger_probe'})
    logger.log_metrics({'probe/lightning': .25}, step=0)
    logger.save()
    logger.finalize('success')
    assert EventAccumulator(logger.log_dir).Reload().Scalars('probe/lightning')[0].value == .25
    child = subprocess.check_output([sys.executable, '-c',
        "import os,sys,json; from tensorboard.compat import tf; tf.io.gfile; "
        "print(json.dumps(dict(flag=os.environ.get('P3_TENSORBOARD_NO_TF'),"
        "hook='tensorboard.compat.notf' in sys.modules, backend=tf.__name__)))"], text=True)
    inheritance = json.loads(child)
    assert inheritance == dict(flag='1', hook=True, backend=backend)
    assert 'tensorflow' not in sys.modules and 'jax' not in sys.modules
    report = dict(author='PhuongTU', passed=True, tensorboard_backend=backend,
                  numpy=numpy.__version__, torch=torch.__version__,
                  scalar_and_grouped_event_write_read='passed', lightning_logger='passed',
                  descendant_inheritance=inheritance, real_tensorflow_imported=False, jax_imported=False)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
