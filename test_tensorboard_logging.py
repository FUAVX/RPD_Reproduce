"""Check real TensorBoard writes using the no-TensorFlow compatibility backend."""

import tempfile
from pathlib import Path

from utils.tensorboard_compat import use_tensorboard_without_tensorflow


def test_real_tensorboard_logging_without_tensorflow():
    # Do not install an import blocker: Lightning/Transformers/PyTorch may probe
    # optional modules with find_spec without actually importing those modules.
    use_tensorboard_without_tensorflow()
    use_tensorboard_without_tensorflow()  # Safe to call more than once.
    from pytorch_lightning.loggers import TensorBoardLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat import tf

    # Resolve the lazy backend and verify TensorBoard selected its own stub.
    assert tf.io.gfile.__name__ == 'tensorboard.compat.tensorflow_stub.io.gfile'
    assert tf.__name__ == 'tensorboard.compat.tensorflow_stub'
    with tempfile.TemporaryDirectory() as directory:
        logger = TensorBoardLogger(directory, name='lightning_logs', version=0, default_hp_metric=False)
        logger.log_metrics({'val_weed_iou': 0.25, 'dpp/down1/alpha/mean': 1.0}, step=1)
        logger.save()  # Exercise the lazy writer initialization that failed in run 2.
        logger.finalize('success')
        events = EventAccumulator(logger.log_dir).Reload()
        assert events.Scalars('val_weed_iou')[0].value == 0.25
        assert events.Scalars('dpp/down1/alpha/mean')[0].value == 1.0
        assert list(Path(logger.log_dir).glob('events.out.tfevents.*'))
    print('PASS: real TensorBoardLogger write/flush/read with TensorBoard tensorflow_stub backend')


if __name__ == '__main__':
    test_real_tensorboard_logging_without_tensorflow()
